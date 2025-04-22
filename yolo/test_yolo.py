import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 pre-trained model (COCO dataset, bao gồm "hand")
model = YOLO("C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\Ego_Hand\\egoyolo_v2\\sign_language_yolo_v22\\weights\\best.pt")  # Bạn có thể thay bằng yolov8s.pt, yolov8m.pt nếu cần chính xác hơn

# Đọc ảnh
image_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\CV\\project\\sign_language_data\\dataa\\test\\B\\15.jpg"  # Thay bằng đường dẫn ảnh của bạn
image = cv2.imread(image_path)

# Dự đoán
results = model(image)

# Lọc ra bounding box có class là "hand"
best_box = None
max_conf = 0

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])  # Lấy class ID
        class_name = model.names[class_id]  # Lấy tên class
        conf = box.conf[0].item()  # Độ tin cậy

        if class_name == "hand" and conf > max_conf:  # Chỉ lấy "hand" có độ tin cậy cao nhất
            max_conf = conf
            best_box = box

# Vẽ bounding box nếu có tay trong ảnh
if best_box is not None:
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])  # Lấy tọa độ bbox
    label = f"Hand: {max_conf:.2f}"  # Ghi nhãn

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bbox
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Hand Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
