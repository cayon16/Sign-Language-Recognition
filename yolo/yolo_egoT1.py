

from ultralytics import YOLO




if __name__ == "__main__": 
    # Load YOLOv8 pre-trained model
    model = YOLO("yolov8n.pt")  # Chọn YOLOv8 Nano (nhẹ nhất)

    # Train model
    results = model.train(
        data="C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\Ego_Hand\\egoyolo_v2\\data.yaml",  
        epochs=50,  
        imgsz=640,  
        batch=32,  
        device="cuda",  
        project="C:\\Users\\ADMIN\\OneDrive\\Desktop\\python\\Ego_Hand\\egoyolo_v3",  # Đặt thư mục lưu kết quả
        name="sign_language_yolo_v2",  # Tạo thư mục riêng để tránh ghi đè
        resume=False  # Tiếp tục train từ checkpoint nếu có
    )

    # Xuất model sau khi train xong
    best_model_path = "runs/train/sign_language_yolo_v3/weights/best.pt"  
    trained_model = YOLO(best_model_path)
    trained_model.export(format="onnx")  # Xuất sang ONNX
    
    
    metrics = trained_model.val()
    print(metrics)
