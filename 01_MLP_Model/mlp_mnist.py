import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Train set size : {len(train_set)} image")
print(f"Test  set size : {len(test_set)} image ")


import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP,self).__init__()
        self.fc1 = nn.Linear(784,128)

        self.fc2 = nn.Linear(128,64)

        self.fc3= nn.Linear(64,10)
    
    def forward(self,x):
        x=x.view(-1,28*28)

        x=F.relu(self.fc1(x))

        x=F.relu(self.fc2(x))
        
        x=F.relu(self.fc3(x))
        return x
    
device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model= SimpleMLP().to(device)

print(model)
        