import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(500),
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "../vrdl_hw1/data" ## put the correct path here
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform["val"])
idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}

model = models.resnext50_32x4d(weights='DEFAULT')

# Modify the fully connected layer to match the number of classes (100)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 100)
)

model.load_state_dict(torch.load("resnext_best.pth", map_location=device))
model.to(device)
model.eval()

image_list = sorted(os.listdir(os.path.join(data_dir, 'test')))
predictions = []

for img_name in image_list:
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")): 
        img_path = os.path.join(os.path.join(data_dir, 'test'), img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform['val'](image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(image)
            pred_label = torch.argmax(output, dim=1).item()

        predictions.append([os.path.splitext(img_name)[0], idx_to_class[pred_label]])

output_csv = "prediction.csv"
df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
df.to_csv(output_csv, index=False)
print("Prediction finish")

