# Object Detection with Rotated Bounding Boxes and Dimension Measurement

This project implements real-time object detection using YOLOv8 and OpenCV, featuring rotated bounding boxes and real-time dimension measurements. The system can detect objects, calculate their dimensions in centimeters, and provide enhanced visualization with rotation-aware bounding boxes.

## Features

- Real-time object detection using YOLOv8
- Rotated bounding box detection
- Dimension measurements in centimeters
- Enhanced image processing for better edge detection
- Confidence threshold filtering
- On-screen display of measurements and confidence scores
- Real-time webcam processing

## Requirements

```bash
python >= 3.8
opencv-python
numpy
ultralytics
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection.git
cd object-detection
```

2. Install required packages:
```bash
pip install opencv-python numpy ultralytics
```

3. Download or prepare your YOLOv8 model weights and update the model path in the code.

## Project Structure

```
├── object_detection.py     # Main script with all implementations
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

## Key Components

### Image Processing Functions

- `rescaleFrame()`: Resizes input frames for processing
- `enhance_image()`: Enhances image quality using CLAHE
- `get_binary_mask()`: Creates binary mask for object segmentation
- `get_accurate_rotated_bbox()`: Calculates rotated bounding boxes

### Measurement Functions

- `pixel_to_cm()`: Converts pixel measurements to centimeters
- `calibrate_focal_length()`: Calibrates camera focal length
- `process_frame()`: Main processing pipeline for object detection and measurement

## Usage

1. Update the YOLO model path in main():
```python
model = YOLO('path_to_your_model.pt')
```

2. Run the script:
```bash
python object_detection.py
```

3. Press 'q' to exit the program

## Calibration

For accurate measurements:

1. Uncomment the calibration section in main()
2. Use a reference object of known dimensions
3. Adjust the focal_length_pixels value
4. Modify depth_cm parameter based on your setup

## Best Practices

For optimal results:

- Ensure good lighting conditions
- Position camera parallel to object plane
- Maintain consistent depth for measurements
- Calibrate system with known reference objects
- Use appropriate confidence thresholds

## Limitations

- Accuracy depends on camera calibration
- Fixed depth assumption (100cm default)
- Requires consistent lighting
- Performance depends on hardware capabilities

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO by Ultralytics
- OpenCV community
- NumPy developers

## Contact

For questions or feedback, please open an issue in the repository.