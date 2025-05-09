# Kidney Vasculature Segmentation with U-Net

## Introduction

This project aims to automatically segment kidney vasculature from high-resolution 3D images obtained through Hierarchical Phase-Contrast Tomography (HiP-CT). By leveraging U-Net architecture, the model efficiently detects vascular structures, filling in gaps left by manual annotation methods. This segmentation can enhance the Vascular Common Coordinate Framework (VCCF) and Human Reference Atlas (HRA), supporting medical research and improving our understanding of how vasculature impacts health.

### Motivation

Manually tracing vascular structures is time-consuming and may take over 6 months for each dataset. Automating the segmentation process with machine learning accelerates data processing, enabling researchers to map cellular relationships across human organs. This project addresses challenges in generalization due to anatomical variability and changing imaging quality.

## Dataset Description

The dataset includes high-resolution 3D kidney images and segmentation masks:

* **Training Set:**

  * TIFF scans of kidneys (50µm and 5.2µm resolution) with segmentation masks.
  * Multiple datasets (e.g., kidney\_1\_dense, kidney\_2) with varying segmentation densities.
* **Test Set:**

  * New kidney images without labeled masks, intended for prediction.
* **train\_rles.csv:** Encoded masks for the training set.

### Directory Structure

```
train/
├── kidney_1_dense/
│   ├── images/
│   └── labels/
test/
├── kidney_5/
├── kidney_6/
train_rles.csv
sample_submission.csv
```

## Model Architecture

The model uses a U-Net architecture implemented with Keras and TensorFlow. Key features include:

* **Image Preprocessing:** Normalization and resizing to 256x256.
* **Data Augmentation:** Rotation, width shifting, and zooming to increase robustness.
* **Loss Function:** Binary Crossentropy with Adam optimizer.
* **Metrics:** Intersection over Union (IoU) and accuracy.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/brunoclbr/EngineeringAI.git
cd EngineeringAI/Kaggle_Kidney_Segmentation
pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python kidney_segmentation_unet.py
```

To evaluate the model:

```bash
python evaluate.py --model model_150sp_8epch.keras
```

## Results

The model is able to identified trends showing a promising performance on test data, evaluated using IoU and accuracy metrics. However, due to lack of computational power no meaningful results were computed.

## Visualization

The script generates plots for training accuracy, loss, and example predictions:

* Training vs Validation Loss
* Training vs Validation Accuracy
* Predicted vs Ground Truth Segmentation

## Contributions

This project contributes to the automation of vascular segmentation in medical imaging, enhancing the ability to map vascular structures efficiently and accurately. This innovation is critical for the development of comprehensive cellular atlases.

## License

[This project is a Kaggle competition project](https://www.kaggle.com/competitions/blood-vessel-segmentation/)
