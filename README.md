# Yolov6QuickUtilTools

# YOLOv6 Quick Utility Tools

A collection of utility scripts designed to streamline the object detection workflow around YOLOv6. This repository focuses on dataset preparation, annotation validation, dataset quality assurance, preprocessing verification, and inference utilities.

The tools were developed to solve common issues encountered during real-world computer vision projects, including annotation conversion, label validation, dataset cleanup, negative sample handling, and model deployment testing.

## Features

### Dataset Validation

* Verify YOLO label format integrity.
* Detect empty or malformed annotation files.
* Validate bounding box coordinate ranges.
* Check class ID consistency.
* Analyze minimum and maximum bounding box dimensions.
* Identify missing image-label pairs.

### Annotation Conversion

* Convert Pascal VOC XML annotations to YOLO format.
* Automatic class mapping support.
* Batch processing of annotation files.

### Dataset Management

* Generate empty labels for negative samples.
* Bulk update class IDs across datasets.
* Detect dataset inconsistencies before training.

### Dataset Quality Assurance

* OpenCV-based batch review tool.
* Visualize bounding boxes directly on images.
* Quarantine problematic image-label pairs.
* Rapid manual inspection workflow.

### Preprocessing Debugging

* Recreate YOLO letterbox transformations.
* Verify bounding box alignment after resizing and padding.
* Visualize exactly what the model receives during training.

### Environment Verification

* Check PyTorch installation.
* Verify CUDA availability.
* Confirm AMP support for mixed-precision training.

### Inference Utilities

* Run YOLOv6 inference on video files.
* FPS monitoring.
* Non-Max Suppression (NMS) integration.
* Output video generation with annotated detections.

## Included Tools

| Script Category             | Purpose                                       |
| --------------------------- | --------------------------------------------- |
| XML to YOLO Converter       | Convert Pascal VOC annotations to YOLO format |
| Label Validator             | Detect invalid annotation files               |
| Class ID Updater            | Bulk relabel datasets                         |
| Dataset Consistency Checker | Find missing image-label pairs                |
| Negative Label Generator    | Create empty labels for negative samples      |
| Dataset Review Tool         | Visual inspection and quarantine workflow     |
| Letterbox Visualizer        | Debug YOLO preprocessing                      |
| Environment Checker         | Verify PyTorch/CUDA setup                     |
| Video Inference Tool        | Run trained YOLOv6 models on videos           |

## Typical Workflow

Dataset Collection

↓

Annotation Conversion

↓

Dataset Validation

↓

Negative Sample Generation

↓

Visual Inspection & Cleanup

↓

Training

↓

Inference & Evaluation

## Requirements

* Python 3.9+
* OpenCV
* PyTorch
* NumPy
* YOLOv6
* PyYAML

Install dependencies:

```bash
pip install opencv-python torch numpy pyyaml
```

## Disclaimer

These utilities were created to support experimentation and training workflows around YOLOv6. They are intended as development and research tools and may require adaptation for production environments.
