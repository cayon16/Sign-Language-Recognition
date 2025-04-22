import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO
import time
# ------------------ CONFIG ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ LOAD YOLO ------------------
def load_yolo_model():
    model = YOLO("C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\Ego_Hand\\egoyolo_v3\\sign_language_yolo_v22\\weights\\best.pt")
    return model.to(device)

yolo_model = load_yolo_model()

# ------------------ HAND DETECTION ------------------
def detect_hand(image):
    results = yolo_model(image)
    best_hand = None
    best_conf = 0
    best_box = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if cls == 0 and conf > 0.3:
                hand_crop = image[y1:y2, x1:x2]
                if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                    if conf > best_conf:
                        best_conf = conf
                        best_hand = cv2.resize(hand_crop, (128, 128))
                        best_box = (x1, y1, x2, y2)  # Lưu bounding box
    return best_hand, best_box


# ------------------ CNN MODEL ------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=24):
        super(CNNModel, self).__init__()
        
        # Load ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------ LOAD MODEL ------------------
model = CNNModel(num_classes=24).to(device)

checkpoint = torch.load("sign_language_model_resnet_v2.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Sign Language Model Loaded!")

# ------------------ TRANSFORM ------------------
valid_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ INFERENCE FUNCTION ------------------
def predict_image(image):
    hand_crop, hand_box = detect_hand(image)

    if hand_crop is None:
        print("Không phát hiện được bàn tay!")
        return

    # Vẽ bounding box lên ảnh
    x1, y1, x2, y2 = hand_box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật (x1, y1, x2, y2)

    hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)

    # Preprocess
    hand_crop = valid_transforms(hand_crop)
    hand_crop = hand_crop.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(hand_crop)
        predicted = torch.argmax(output, dim=1).item()

    # Convert label to letter
    if predicted >= 9:
        label = chr(predicted + ord('A') + 1)
    else:
        label = chr(predicted + ord('A'))

    print(f"Kết quả dự đoán: {label} ({predicted})")

    # Hiển thị ảnh và kết quả
    cv2.putText(image, f"Predicted: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị ảnh video
    cv2.imshow("Live Sign Language Prediction", image)


# ------------------ RUN VIDEO ------------------
def run_video():
    cap = cv2.VideoCapture(0)  # Mở webcam (hoặc video file nếu cần)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        predict_image(frame)  # Dự đoán ảnh

        # Hiển thị ảnh
        cv2.imshow("Live Sign Language Prediction", frame)

        # Dừng video khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    run_video()