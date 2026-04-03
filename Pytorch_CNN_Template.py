import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Grayscale images have 1 channel, If you are using RGB images, set this to 3
# This is the sizes of each layer in the CNN architecture, you can modify these as needed for your specific task.

Input_Layer_1 = 1 
Output_Layer_1 = 32
Output_Layer_2 = 64

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10

    # Device configuration, This will use GPU if available, otherwise it will fall back to CPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # This is the Dataset for MNIST, you can replace it with your own dataset and adjust the transform accordingly  

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model definition (simple CNN)
    # This is a CNN ARchitecture it defines two convolutional layers followed by two fully connected layers.
    # You can modify this architecture as needed for your specific task. 

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(Input_Layer_1, Output_Layer_1, kernel_size=3)
            self.conv2 = nn.Conv2d(Output_Layer_1, Output_Layer_2, kernel_size=3)
            self.fc1 = nn.Linear(Output_Layer_2 * 12 * 12, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 64 * 12 * 12)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)

    # Loss and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop

    for epoch in range(num_epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        scheduler.step()

# The architeture and setup is all in a funtion called main. Running this final two lines will execute the main function and start the training process.
# For more advanced tasks you may want to add more layers 
# Good luck with your training, and feel free to modify the architecture, hyperparameters, and dataset as needed for your specific task!

if __name__ == '__main__':
    main()
