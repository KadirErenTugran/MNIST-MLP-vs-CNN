import torch
from torchvision import datasets,transforms
import torch.optim as optim
from  torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

print(f"Train set size : {len(train_set)} image")
print(f"Test set size : {len(test_set)} image ")


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
        
        x=self.fc3(x)
        return x


device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model= SimpleMLP().to(device)
print(device,"running...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)  #lr=> Learning Rate

epochs = 5
train_losses = []

print("training is starting...")

for epochs in range(epochs):
    running_loss=0.0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
    
    avg_loss=running_loss/len(train_loader)
    train_losses.append(avg_loss)
    print(f"epochs => {epochs} done  MSE=>{avg_loss:.4f}")

correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device), labels.to(device)

        outputs=model(images)

        _,predicted = torch.max(outputs.data, 1)
        total+= labels.size(0)
        correct+=(predicted==labels).sum().item()
accuracy=100 * correct/total

print(f"Model accuracy on 10,000 test images: %{accuracy:.2f}") # %96.42 accuracy

print(model)
        