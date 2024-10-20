import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5l6') 

img = cv2.imread('1.jpg') 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

results = model(img)

results.print() 

results.render() 

cv2.imshow('YOLOv5 Object Detection', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


object_count = len(results.xyxy[0]) 
print(f'تعداد اشیاء شناسایی‌شده: {object_count}')
