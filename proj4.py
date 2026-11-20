import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transfrm import VisionTransformer
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def accuracy(model,loader):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            test_logits,_ = model(x)
            preds = torch.argmax(test_logits,dim = 1)
            correct += (preds==y).sum().item()
            total += y.shape[0]
    return correct/total

def visualize_attention(img, attn_weights, axes):

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    original = img.cpu().permute(1, 2, 0).numpy()
    original = np.clip(original * std + mean, 0, 1)

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, attn in enumerate(attn_weights):
        attn = attn.reshape(1,65,65)
        weights = attn[0, 0, 1:].detach().cpu()
        weights = weights.reshape(1, 1, 8, 8)

        weights = torch.nn.functional.interpolate(
            weights,
            size=(32, 32),
            mode="bilinear",
            align_corners=False
        )[0, 0].numpy()

        weights = (weights - weights.min()) / (
            weights.max() - weights.min() + 1e-8
        )

        axes[i + 1].imshow(original)
        axes[i + 1].imshow(weights, cmap="jet", alpha=0.45)
        axes[i + 1].set_title(f"head {i + 1}")
        axes[i + 1].axis("off")

device = "cuda" if torch.cuda.is_available() else "cpu"
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
])
eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))])

train_data = datasets.CIFAR10(root = "./cifar_data" , train = True ,download=False, transform=train_tf)
train_eval_data = datasets.CIFAR10(root = "./cifar_data",train=True,download = False,transform=eval_tf)
test_data = datasets.CIFAR10(root = "./cifar_data" , train = False ,download= False , transform=eval_tf)

train_loader = DataLoader(train_data,64,True)
train_eval_loader = DataLoader(train_eval_data,64,False)
test_loader = DataLoader(test_data,64,False)

model = VisionTransformer(32,4,3,128,4,512,4,10)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters() , lr =5e-4 )

if not os.path.exists("vsn_transfrm_v1.pth"):
    with torch.no_grad():
        model.eval()
        x,y = next(iter(train_eval_loader))
        x = x.to(device)
        y = y.to(device)
        logits,_ = model(x)
        loss = criterion(logits,y)
    print(f"Iniital training loss : {loss:.4f}")
    for i in range(30):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad()
            logits,_ = model(x)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
        train_acc = accuracy(model,train_eval_loader)
        test_acc = accuracy(model,test_loader)
        print(f"Epoch {i} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

    torch.save(model.state_dict(),"vsn_transfrm_v1.pth")
else:
    state_dict = torch.load("vsn_transfrm_v1.pth")
    model.load_state_dict(state_dict)
    model.eval()

test_acc = accuracy(model,test_loader)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print(f"Test Accuracy: {test_acc:.4f}")
classes = train_data.classes

indices = np.random.choice(len(train_eval_data), 8, replace=False)
index = np.random.randint(len(train_eval_data))
fig, axes = plt.subplots(4, 5, figsize=(15, 24))

model.eval()

with torch.no_grad():
    img, label = train_eval_data[index]
    
    logits, attn_weights = model(
        img.unsqueeze(0).to(device)
    )
    
    pred = classes[logits.argmax(dim=1).item()]
    for row in range(4):
        visualize_attention(
            img,
            attn_weights[row][0],
            axes[row]
        )

        axes[row, 0].set_title(
            f"True: {classes[label]} | Pred: {pred}"
        )

plt.tight_layout()
plt.show()



