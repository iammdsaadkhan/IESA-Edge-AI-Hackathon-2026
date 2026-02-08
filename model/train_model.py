import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Transforms
# --------------------
train_tfms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_tfms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --------------------
# Dataset
# --------------------
train_ds = datasets.ImageFolder("dataset/train", transform=train_tfms)
val_ds   = datasets.ImageFolder("dataset/validation", transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

class_names = train_ds.classes

# --------------------
# Model
# --------------------
class SEMNet_Final_1_5M(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SEMNet_Final_1_5M(len(class_names)).to(device)

# --------------------
# Class-balanced loss
# --------------------
counts = Counter(train_ds.targets)
weights = torch.tensor([1.0 / counts[i] for i in range(len(counts))]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

# --------------------
# Train / Eval
# --------------------
def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total


# --------------------
# Training loop
# --------------------
EPOCHS = 30
best_val = 0
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    tl, ta = run_epoch(model, train_loader, True)
    vl, va = run_epoch(model, val_loader, False)

    scheduler.step(va)

    history["train_acc"].append(ta)
    history["val_acc"].append(va)
    history["train_loss"].append(tl)
    history["val_loss"].append(vl)

    if va > best_val:
        best_val = va
        torch.save(model.state_dict(), "models/best_model.pth")

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {ta*100:.2f}% | Val: {va*100:.2f}%")

# --------------------
# Reports
# --------------------
os.makedirs("reports", exist_ok=True)

# Accuracy plot
plt.plot(history["train_acc"], label="Train")
plt.plot(history["val_acc"], label="Val")
plt.legend()
plt.savefig("reports/training_history.png")
plt.close()

# Evaluation
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in val_loader:
        out = model(x.to(device))
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(y.numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.savefig("reports/confusion_matrix.png")
plt.close()

report = classification_report(y_true, y_pred, target_names=class_names)
with open("reports/model_metrics_report.txt", "w") as f:
    f.write(report)

with open("reports/metrics.json", "w") as f:
    json.dump({
        "accuracy": best_val,
        "epochs": EPOCHS,
        "classes": class_names
    }, f, indent=2)
