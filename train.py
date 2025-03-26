import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is ", device)

transform = {
    "train": transforms.Compose(
        [
            transforms.Resize(600),
            transforms.RandomCrop(500),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(600),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

data_dir = "../vrdl_hw1/data"  # put the correct path here
train_dataset = datasets.ImageFolder(
    root=f"{data_dir}/train", transform=transform["train"]
)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform["val"])
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models.resnext50_32x4d(weights="DEFAULT")

# Modify the fully connected layer to match the number of classes (100)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 100))

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00025, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.1, patience=5, verbose=True
)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    current_lr = optimizer.param_groups[0]["lr"]
    train_acc = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}], lr: {current_lr:.6f}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%"
    )

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "resnext_best.pth")
        print(f"New best model saved with accuracy: {best_val_acc:.2f}%")
print("Training Process done!")
