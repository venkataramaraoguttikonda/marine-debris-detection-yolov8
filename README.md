# Marine Debris Detection using YOLOv8 on Satellite Imagery

This repository presents a modular object detection pipeline for identifying **marine debris** in satellite images using **YOLOv8**.  
We fine-tune a YOLOv8m model on geospatial satellite data, enabling detection of floating debris and similar marine pollutants in real-world ocean conditions.

The pipeline includes:
- Inference using YOLOv8m model
- Bounding box evaluation (IoU, precision, recall, F1-score)
- Visualization of predicted and ground-truth boxes on test images

## Table of Contents

- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Results & Performance](#results--performance)
  - [Model Training Curves](#model-training-curves)
  - [Model Predictions vs Ground Truth](#model-predictions-vs-ground-truth)
- [Folder Structure](#folder-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used is from the [NASA-IMPACT Marine Debris ML](https://github.com/NASA-IMPACT/marine_debris_ML) project, containing satellite imagery with YOLO-format bounding box annotations for `marine debris`.

It includes three standard splits:

```
data_yolo/
├── train/
├── val/
└── test/
    ├── images/      # Input satellite image tiles (.jpg)
    └── labels/      # Corresponding YOLO annotations (.txt)
```

Each `.txt` label file follows the YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

> Note: This repository only includes the **test set** for evaluating our best models.  
> For full dataset access (train/val), refer to the original NASA-IMPACT repo linked above.

---

#### Dataset Attribution

**Note:** This dataset and its annotations were directly sourced from the official NASA-IMPACT project:

> **Marine debris detection with commercial satellite imagery and deep learning**  
> GitHub: [https://github.com/NASA-IMPACT/marine_debris_ML](https://github.com/NASA-IMPACT/marine_debris_ML)  
> © NASA IMPACT Project – All credit for data curation and annotation belongs to the original authors.

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/venkataramaraoguttikonda/marine-debris-detection-yolov8.git
cd marine-debris-detection-yolov8
```

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate yolo
```

**To train the model from scratch:**

```bash
python src/train_yolo.py
```

This downloads `yolov8m.pt` and saves the trained model to:

```
runs/detect/yolov8m_b16/weights/best.pt
```

**The Best trained model is already available at the above path.**

To evaluate that model:

```bash
python src/eval_yolo.py
```

To visualize the model predictions:

```bash
python src/vis_yolo.py
```

Outputs are saved to:

```
vis_outputs/
```
## Results & Performance

The final model was trained for 100 epochs with YOLOv8m (`yolov8m.pt`) at 320×320 resolution using a confidence threshold of 0.1 and IoU threshold of 0.3.


| Metric           | Value   |
|------------------|---------|
| True Positives   | 57      |
| False Positives  | 35      |
| False Negatives  | 36      |
| Accuracy         | 0.4453  |
| Precision        | 0.6196  |
| Recall           | 0.6129  |
| F1-score         | 0.6162  |


---

#### Model Training Curves

The following plots were generated during training and are saved in the `runs/` directory:

- **Loss and mAP curves:**  
  ![results](runs/detect/yolov8m_b16/results.png)

**F1 vs Confidence | Precision vs Confidence | Recall vs Confidence | Precision-Recall Curve:**
<p align="center">
  <img src="runs/detect/yolov8m_b16/F1_curve.png" alt="F1 Curve" width="24%"/>
  <img src="runs/detect/yolov8m_b16/P_curve.png" alt="Precision Curve" width="24%"/>
  <img src="runs/detect/yolov8m_b16/R_curve.png" alt="Recall Curve" width="24%"/>
  <img src="runs/detect/yolov8m_b16/PR_curve.png" alt="PR Curve" width="24%"/>
</p>

**Label distribution and correlation:**
<p align="center">
  <img src="runs/detect/yolov8m_b16/labels.jpg" alt="Label Distribution" width="49%"/>
  <img src="runs/detect/yolov8m_b16/labels_correlogram.jpg" alt="Label Correlogram" width="49%"/>
</p>

---

#### Model Predictions vs Ground Truth

Three sample predictions from the `vis_outputs/` folder are visualized below:

<p align="center">
  <img src="runs/detect/yolov8m_b16/20170323_152750_0e20_17297-29802-16.jpg" alt="Sample 1" width="32%"/>
  <img src="runs/detect/yolov8m_b16/20170326_153233_0e26_17057-29746-16.jpg" alt="Sample 2" width="32%"/>
  <img src="runs/detect/yolov8m_b16/20180313_154258_1008_17028-29718-16.jpg" alt="Sample 3" width="32%"/>
</p>

Green: ground truth  
Red: YOLOv8 predictions


## Folder Structure

```
.
├── config_yolo2.yaml              # YOLO dataset config file
├── environment_yolo.yaml          # Conda environment definition
├── yolov8m.pt                     # Base pretrained YOLOv8 model
├── data_yolo/
│   └── test/
│       ├── images/                # Test satellite images
│       └── labels/                # YOLO-format ground truth labels
├── runs/
│   └── detect/
│       └── yolov8m_b16/           # Training logs and model outputs
│           ├── weights/           # Contains best.pt
│           ├── results.png        # Loss & metric plots
│           ├── F1_curve.png       # F1 vs confidence
│           ├── PR_curve.png       # Precision-Recall curve
│           └── ...
├── vis_outputs/                   # Output images from vis_yolo.py
├── src/
│   ├── config_yolo.py
│   ├── train_yolo.py
│   ├── eval_yolo.py
│   ├── vis_yolo.py
│   └── utils_yolo.py
```

---

## Citation

If you use this repository or parts of it in your research or project, please cite:

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- NASA-IMPACT Marine Debris ML: https://github.com/NASA-IMPACT/marine_debris_ML

---

## Acknowledgements

This work was developed as part of the **Deep Learning for Computer Vision (DS265)** course at the **Indian Institute of Science (IISc), Bengaluru**.

**Course Instructor:** Prof. R. Venkatesh Babu

**Contributors:**
- Venkata Ramarao Guttikonda (M.Tech – CAOS, IISc)
- Ananya Krishna Devupalli (M.Tech – CAOS, IISc)

Special thanks to:
- **NASA IMPACT** for the open-source MARIDA dataset
- The open-source communities behind **PyTorch**, **Hugging Face Transformers**, and **Ultralytics YOLO**
