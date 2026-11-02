import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data  import DataLoader

def accuracy(model,loader):
    model.eval()
    correct = 0
    total = 0
    for x,y in loader:
        with torch.no_grad():
            test_logits = model(x)
            preds = torch.argmax(test_logits,dim = 1)
            correct += (preds==y).sum().item()
            total += y.shape[0]
    return correct/total

class CIFAR_CNN(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.layers = nn.ModuleList()

        self.no_of_layers = len(sizes) - 1
        for i in range(self.no_of_layers-1):
            self.layers.append(nn.ModuleList([nn.Conv2d(in_channels=sizes[i],out_channels = sizes[i+1],kernel_size=3,padding = 1),nn.MaxPool2d(kernel_size=2,stride = 2)]))
        self.layers.append(nn.LazyLinear(out_features=sizes[-1]))
    def forward(self,x):
        inp = x
        for i in range(self.no_of_layers-1):
            z = self.layers[i][0](inp)
            a = torch.relu(z)
            inp = self.layers[i][1](a)
        z =self.layers[-1](torch.flatten(inp,1,3))
        return z

sizes = [3,32,64,128,10]

transform = transforms.ToTensor()

train_data = datasets.CIFAR10(root = "./cifar_data",train = True , download=False , transform=transform)
test_data = datasets.CIFAR10(root = "./cifar_data",train = False,download=False , transform = transform)

train_loader = DataLoader(train_data,batch_size = 64,shuffle = True) #(Image , Label)
test_loader = DataLoader(test_data,batch_size = 64 , shuffle = True)

model = CIFAR_CNN(sizes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr = 1e-3)

for i in range(20):
    model.train()
    for x,y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()

    train_acc = accuracy(model,train_loader)
    test_acc = accuracy(model,test_loader)
    print(f"Epoch {i} | Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")