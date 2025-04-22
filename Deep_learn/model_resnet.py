import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO model
def load_yolo_model():
    model = YOLO("C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\Ego_Hand\\egoyolo_v3\\sign_language_yolo_v22\\weights\\best.pt")
    return model.to(device)

yolo_model = load_yolo_model()

def detect_hand(image):
    results = yolo_model(image)
    best_hand = None
    best_conf = 0  # Độ tin cậy cao nhất

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Chuyển về float
            cls = int(box.cls[0])  # Lớp dự đoán (0 là bàn tay)

            if cls == 0 and conf > 0.3:
                hand_crop = image[y1:y2, x1:x2]
                if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                    if conf > best_conf:  # Chỉ giữ lại tay có độ tin cậy cao nhất
                        best_conf = conf
                        best_hand = cv2.resize(hand_crop, (128, 128))
                        
    return best_hand

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory)]
        for f in file_names:
            img = cv2.imread(f)
            img = detect_hand(img)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if ord(d) - ord('A') >9:
                    labels.append(ord(d) - ord('A') - 1)
                else:
                    labels.append(ord(d) - ord('A'))  
            
    return np.array(images), np.array(labels)

ROOT_PATH_1 = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\CV\\project\\sign_language_data\\dataa\\asl_alphabet_train"
train_data_directory_1 = os.path.join(ROOT_PATH_1, "asl_alphabet_train")
images_1, labels_1 = load_data(train_data_directory_1)

ROOT_PATH_2 = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\CV\\project\\sign_language_data\\dataa\\"
train_data_directory_2 = os.path.join(ROOT_PATH_2, "train")
images_2, labels_2 = load_data(train_data_directory_2)

ROOT_PATH_3 = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\CV\\project\\sign_language_data\\dataa\\"
train_data_directory_3 = os.path.join(ROOT_PATH_3, "train_2")
images_3, labels_3 = load_data(train_data_directory_3)

ROOT_PATH_4 = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\CV\\project\\sign_language_data\\dataa\\"
train_data_directory_4 = os.path.join(ROOT_PATH_4, "train_3")
images_4, labels_4 = load_data(train_data_directory_4)

ROOT_PATH_TEST_1 = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\CV\\project\\sign_language_data\\dataa\\"
test_data_directory_1 = os.path.join(ROOT_PATH_TEST_1, "test")
images_test_1, labels_test_1 = load_data(test_data_directory_1)

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(images_2, labels_2, test_size=0.2, random_state=42)
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(images_3, labels_3, test_size=0.2, random_state=42)

x_train = np.concatenate((images_1, x_train_2, x_train_3, images_4), axis=0)
y_train = np.concatenate((labels_1, y_train_2, y_train_3, labels_4), axis=0)

x_valid = np.concatenate((x_test_2, x_test_3, images_test_1), axis=0)
y_valid = np.concatenate((y_test_2, y_test_3, labels_test_1), axis=0)


from collections import Counter
# Với train
train_counter = Counter(y_train)
print("Số lượng ảnh theo label trong tập train:")
for label, count in sorted(train_counter.items()):
    print(f"Label {label}: {count} ảnh")

# Với val
val_counter = Counter(y_valid)
print("\nSố lượng ảnh theo label trong tập val:")
for label, count in sorted(val_counter.items()):
    print(f"Label {label}: {count} ảnh")

# Chuyển sang tensor
y_train = torch.tensor(y_train, dtype=torch.long)
y_valid = torch.tensor(y_valid, dtype=torch.long)


# Define data augmentation transformations
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

    
class SignLanguageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # Đảm bảo rằng self.images và self.labels được khởi tạo đúng cách
        self.images = images  # Không cần chuyển sang np.float32
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)  # Lấy chiều dài từ images
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

dataset_train = SignLanguageDataset(x_train, y_train, transform=train_transforms)
dataset_valid = SignLanguageDataset(x_valid, y_valid, transform=valid_transforms)
train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=64, shuffle=False)



def show_samples(loader, title, num_samples=5):
    data_iter = iter(loader)
    images, labels = next(data_iter)  # Lấy batch đầu tiên
    
    # Lấy num_samples ảnh ngẫu nhiên từ batch
    num_samples = min(num_samples, len(images))  # Đảm bảo không lấy quá số lượng ảnh trong batch
    indices = torch.randperm(len(images))[:num_samples]  # Lấy num_samples chỉ số ngẫu nhiên
    images = images[indices]  # Lấy các ảnh ngẫu nhiên
    labels = labels[indices]  # Lấy các nhãn tương ứng

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle(title, fontsize=14)

    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).cpu().numpy()  # Chuyển từ tensor về numpy
        img = (img - img.min()) / (img.max() - img.min())  # Chuẩn hóa về khoảng [0,1]

        # Chuyển nhãn số thành chữ cái
        if labels[i].item() > 9:
            label = chr(labels[i].item() + ord('A') + 1)  # Điều chỉnh cho các nhãn từ 'K' trở đi
        else:
            label = chr(labels[i].item() + ord('A'))  # Chuyển số thành chữ cái 'A'-'J'

        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")  # Chỉ hiển thị nhãn chữ cái
        axes[i].axis("off")
    
    plt.show()


# Hiển thị ảnh từ train set
show_samples(train_loader, "Train Set Samples")

# Hiển thị ảnh từ validation set
show_samples(valid_loader, "Validation Set Samples")

'''
# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes=24):
        super(CNNModel, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.25)

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.25)

        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout(0.25)
        
        # Fifth Convolutional Block (mới thêm)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # 
        self.dropout5 = nn.Dropout(0.25)

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)  # 128x128 -> 8x8 after 4 poolings
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)

        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        
        x = self.pool(F.relu(self.conv5(x)))
        x = self.dropout4(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        
        return x
'''
# resnet18 finetune 
class CNNModel(nn.Module):
    def __init__(self, num_classes=24):
        super(CNNModel, self).__init__()
        
        # Load ResNet18 backbone, bỏ phần FC cuối
        self.backbone = models.resnet18(pretrained=True)  # Bạn có thể đổi thành pretrained=True nếu muốn dùng trọng số đã huấn luyện sẵn
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Bỏ phần avgpool và fully-connected
        
        # Dropout sau backbone để giảm overfitting
        self.dropout = nn.Dropout(0.25)
        
        # Fully Connected layers
        # Với input ảnh 128x128, sau backbone của ResNet18 ta thu được feature map kích thước (batch, 512, 4, 4)
        # Nên số đầu vào của lớp fc là 512 * 4 * 4 = 8192
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)        # Lấy feature map có kích thước (batch, 512, 4, 4)
        x = torch.flatten(x, start_dim=1)  # Chuyển về vector (batch, 8192)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def calculate_accuracy(outputs, labels):
    if outputs.shape[0] == 0:  # Tránh lỗi với batch rỗng
        return 0, 0
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct, labels.size(0)  # Số lượng dự đoán đúng và tổng số mẫu

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Chuyển về GPU
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        correct, samples = calculate_accuracy(outputs, labels)
        total_correct += correct
        total_samples += samples
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = total_correct / total_samples  # Độ chính xác toàn tập
    
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    val_total_correct = 0
    val_total_samples = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            correct, samples = calculate_accuracy(outputs, labels)
            val_total_correct += correct
            val_total_samples += samples
    
    val_epoch_loss = val_running_loss / len(valid_loader)
    val_epoch_accuracy = val_total_correct / val_total_samples
    
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")

torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_classes': 24}, "sign_language_model_resnet.pth")

