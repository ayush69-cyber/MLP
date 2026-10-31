import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    sigz = sigmoid(z)
    return sigz*(1-sigz)

def relu(z):
    return np.where(z>0,z,0)

def relu_derivative(z):
    return np.where(z>0,1,0)

def layer_forward(x, W, b, activation = None):
    z = np.dot(x,np.transpose(W)) + b
    if activation is not None:
        return(activation(z))
    return z

def network_forward(x,W1,b1,W2,b2):
    a1 = layer_forward(x,W1,b1,relu)
    a2 = layer_forward(a1,W2,b2)
    return a2

def backward_pass(x,y,W1,b1,W2,b2):
    z1 = layer_forward(x,W1,b1)
    a1 = layer_forward(x,W1,b1,relu)
    z2 = layer_forward(a1,W2,b2)
    a2 = z2
    n = W2.shape[0]
    delta2 = 2/n*(a2-y)
    delta1 = np.dot(np.transpose(W2),delta2)*relu_derivative(z1)
    dw1 = np.outer(delta1,x)
    dw2 = np.outer(delta2,a1)
    db1 = delta1
    db2 = delta2
    return dw1,db1,dw2,db2

def compute_loss(x,y,W1,b1,W2,b2):
    a2 = network_forward(x,W1,b1,W2,b2)
    n = W2.shape[0]
    L = (1/n*(y-a2)**2).sum()
    return L

def softmax(z):
    no_of_batches = z.shape[0]
    p = []
    for i in range(no_of_batches):
        p.append(np.exp(z[i]-np.max(z[i]))/np.sum(np.exp(z[i]-np.max(z[i]))))
    return np.array(p)

def ce_loss_and_delta(logits,y):
    batch =  logits.shape[0]
    p = softmax(logits)
    L = -1/batch*np.sum(np.log(p[np.arange(batch),y]))
    delta = 1/batch*p
    delta[np.arange(batch),y] -= 1/batch
    return L , delta 

def init_params(sizes):
    Ws = []
    bs = []
    for i in range(len(sizes)-1):
        Ws.append(np.random.randn(sizes[i+1],sizes[i])*np.sqrt(2/sizes[i]))
        bs.append(np.zeros(sizes[i+1]))  
    return Ws,bs

def forward(X,Ws,bs):
    no_of_layers = len(Ws)
    a_prev = []
    z = []
    temp = X.copy()
    for i in range(no_of_layers-1):
        a_prev.append(temp)
        temp = layer_forward(temp,Ws[i],bs[i])
        z.append(temp)
        temp = relu(temp)
    a_prev.append(temp)
    logits = layer_forward(temp,Ws[no_of_layers-1],bs[no_of_layers-1])
    return a_prev,z,logits

def backward(X,y,Ws,bs):
    a_prev,z,logits = forward(X,Ws,bs)
    dw = []
    db = []
    no_of_layers = len(Ws)
    L,delta = ce_loss_and_delta(logits,y)
    for i in range(no_of_layers-1):
        dw.append(np.dot(np.transpose(delta),a_prev[-1-i]))
        db.append(np.sum(delta,axis = 0))
        delta = np.dot(delta,Ws[-1-i])*relu_derivative(z[-1-i])
    dw.append(np.dot(np.transpose(delta),a_prev[0]))
    db.append(np.sum(delta,axis = 0))
    return L,dw[-1:-1-no_of_layers:-1],db[-1:-1-no_of_layers:-1]

def accuracy(loader):
    correct = 0
    total = 0
    for x,y in loader:
        _,_,test_logits = forward(x.flatten(1,3).__array__(),Ws,bs)
        preds = np.argmax(test_logits,axis = 1)
        correct += (preds==y).sum().item()
        total += y.shape[0]
    return correct/total

def torch_accuracy(model,loader):
    model.eval()
    correct = 0
    total = 0
    for x,y in loader:
        with torch.no_grad():
            test_logits = model(torch.flatten(x,1,3))
            preds = torch.argmax(test_logits,dim = 1)
            correct += (preds==y).sum().item()
            total += y.shape[0]
    return correct/total

def train_step(x,y,W1,b1,W2,b2,lr):
    dw1,db1,dw2,db2 = backward_pass(x,y,W1,b1,W2,b2)
    W1 = W1 - lr*dw1
    b1 = b1 - lr*db1
    W2 = W2 - lr*dw2
    b2 = b2 - lr*db2
    return W1,b1,W2,b2

def init_wts_biases(layer):
    if isinstance(layer,(nn.Linear)):
        nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
        nn.init.zeros_(layer.bias)


transform = transforms.ToTensor()

train_data = datasets.MNIST(root = "./data",train = True , download=False, transform=transform) #[(Tensor,Label)]
test_data = datasets.MNIST(root = "./data",train = False,download=False,transform=transform)

train_loader = DataLoader(train_data,batch_size=64,shuffle = True) #tuple (images/Tensor, labels)
test_loader = DataLoader(test_data,batch_size=64,shuffle= True) #tuple (images/Tensor , labels)

sizes = [784, 128, 64, 10]
Ws,bs = init_params(sizes)
lr = 0.01

k=0
for i in range(10):
    mean = 0
    loss_count = 0
    for x,y in train_loader:
        L,dw,db = backward(x.flatten(1,3).__array__(),torch.flatten(y).__array__(),Ws,bs)
        mean = mean + L
        loss_count += 1
        if k==0:
            print(f"Initial Loss : {L:.4f}")
            k = 1
        for j in range(len(Ws)):
            Ws[j] = Ws[j] - lr*dw[j]
            bs[j] = bs[j] - lr*db[j]
    mean = mean/loss_count
    print(f"Mean_Loss : {mean:.4f}")
    print(f"Train Accuracy : {accuracy(train_loader):.4f}")
    print(f"Test Accuracy : {accuracy(test_loader):.4f}")

# Equivalent torch Implementation

class MLP(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.no_of_layers = len(sizes)-1
        self.layers = nn.ModuleList()
        for i in range(self.no_of_layers):
            self.layers.append(nn.Linear(sizes[i],sizes[i+1]))
    def forward(self,x):
        a = x
        for i in range(self.no_of_layers-1):
            z = self.layers[i](a)
            a = torch.relu(z)
        return self.layers[-1](a)

    def copy_params(self,Ws,bs):
        with torch.no_grad():
            for i in range(self.no_of_layers):
                self.layers[i].weight.copy_(torch.tensor(Ws[i]))
                self.layers[i].bias.copy_(torch.tensor(bs[i]))

model = MLP(sizes)
model.apply(init_wts_biases)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
print("Torch Implementataion.")
for i in range(10):
    model.train()
    for x,y in train_loader:
        optimizer.zero_grad()
        logits = model(torch.flatten(x,1,3))
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
    print(f"Train Accuracy : {torch_accuracy(model,train_loader):.4f}")
    print(f"Test Accuracy : {torch_accuracy(model,test_loader):.4f}")

model.copy_params(Ws,bs)
x,y = next(iter(test_loader))       
_,_,np_logits = forward(x.flatten(1,3).__array__(),Ws,bs)       
model.eval()
torch_logits = model(x.flatten(1,3))
print(np.allclose(np_logits,torch_logits.detach().__array__(),atol = 1e-4))
print(f"Correct Label : {y} | np_network_preds : {np.argmax(np_logits,axis = 1)} | torch_network_preds : {torch.argmax(torch_logits,dim = 1)}")