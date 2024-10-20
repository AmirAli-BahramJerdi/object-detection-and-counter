from ultralytics import YOLO
import cv2

model = YOLO("yolo11s.pt")
image_path = '5.jpg'
img = cv2.imread(image_path)

results = model.predict(img) 
model.train(data="coco8.yaml", epochs=10)
metrics = model.val()
model.export(format="onnx")

for result in results:
    boxes = result.boxes 
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0] 
        conf = box.conf[0] 
        cls = box.cls[0]  
        label = model.names[int(cls)]  
        
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


cv2.imshow('YOLOv8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

object_count = len(results[0].boxes)
print(f'number of objects: {object_count}')
