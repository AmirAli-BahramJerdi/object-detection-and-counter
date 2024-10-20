# Object Detection with YOLO

This project implements object detection using the YOLO (You Only Look Once) algorithm, specifically utilizing the YOLOv5 (or YOLOv8) model. The primary goal of this project is to accurately identify and classify objects in images.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Features

- Real-time object detection
- Easy to use with pre-trained YOLOv5 (or YOLOv8) models
- Visual output with bounding boxes and labels
- Simple integration with OpenCV for image processing

## Installation

To run this project, you'll need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:

   ```bash
   git clone https://github.com/AmirAli-BahramJerdi/object-detection-and-counter.git
   cd object-detection-and-counter
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

To use the object detection model, simply run the script with the desired image:

```bash
python detect.py --image path/to/your/image.jpg
```

You can adjust the confidence threshold and model type as needed.

## Example

Here is a brief example of how to use the model in Python:

```python
import cv2
import torch

# Load YOLOv5 (or YOLOv8) model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Change to 'yolov8' for YOLOv8

# Load an image
image_path = 'path/to/your/image.jpg'
img = cv2.imread(image_path)

# Perform object detection
results = model(img)

# Render results on the image
img_with_boxes = results.render()[0]

# Display the image with detected objects
cv2.imshow('YOLO Detection', img_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
