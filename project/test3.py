import cv2
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

image_path = '5.jpg'
img = cv2.imread(image_path)

window_size = 640 
step_size = 320   

all_results = []

for y in range(0, img.shape[0] - window_size + 1, step_size):
    for x in range(0, img.shape[1] - window_size + 1, step_size):
        window = img[y:y + window_size, x:x + window_size]
        
        results = model(window, conf=0.25, iou=0.45) 
        all_results.append((results, x, y))
        
        object_count = len(results[0].boxes)
        print(f'Objects detected in window at position ({x}, {y}): {object_count}')

for (results, x_offset, y_offset) in all_results:
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = model.names[int(cls)]
            
            x1 += x_offset
            x2 += x_offset
            y1 += y_offset
            y2 += y_offset
            
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imshow('YOLOv8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
