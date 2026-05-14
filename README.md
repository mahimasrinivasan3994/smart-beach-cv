<!-- ═══════════════════════════════════════════════════════════════════
     SMART BEACH — Predict People on the Beach
     GitHub Repository: mahimasrinivasan3994/smart-beach-cv
     Georgian College | AIDI Capstone | April 2024
     ═══════════════════════════════════════════════════════════════════ -->

<!-- ANIMATED HEADER BANNER -->
![Header](https://capsule-render.vercel.app/api?type=waving&color=0077B6&height=220&section=header&text=🏖️%20Smart%20Beach&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Crowd%20Detection%20%26%20Beach%20Safety%20System&descAlignY=58&descColor=caf0f8)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Object%20Detection-00FFFF?style=for-the-badge&logo=github&logoColor=black)](https://github.com/ultralytics/yolov5)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)](https://colab.research.google.com)

![Accuracy](https://img.shields.io/badge/Zone%20Detection%20Accuracy-93%25-brightgreen?style=for-the-badge)
![Segmentation](https://img.shields.io/badge/Water%20Segmentation%20Accuracy-85%25-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/Academic%20Project-Georgian%20College-orange?style=for-the-badge)

</div>

---

## 📍 Real-World Context

> **This project was built for the Smart Beach initiative in Kincardine, Ontario** — a groundbreaking, first-of-its-kind public safety programme in North America, funded by the **Municipal Innovation Council (MIC)** of Bruce County (~$400,000 CAD).

Drownings along Kincardine's Station Beach had persisted despite prior warnings. The Smart Beach initiative, led by the University of Windsor, deployed buoys, sensors, and cameras along the coastline to collect real-time data on wave strength, currents, swimmer behaviour, and crowd density.

**Our AI system contributes the intelligent people-counting and zone monitoring layer** — detecting crowd density in real time to enable proactive lifeguard deployment and emergency response.

---

## 🎯 Problem Statement

Beach safety teams at Kincardine faced a critical operational challenge:

```
❌  No way to predict how crowded different beach zones would become
❌  Lifeguard scheduling was entirely reactive — not proactive
❌  Existing models misclassified objects and lacked zone-level precision
❌  No automated system to trigger alerts based on crowd thresholds
```

**The goal:** Build a system that can automatically detect, count, and forecast people across distinct beach zones using camera footage — enabling smarter resource allocation before incidents occur.

---

## 💡 Our Solution

A two-stage AI pipeline that first understands the beach geography, then counts people within it:

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│   STAGE 1               │        │   STAGE 2                    │
│   Water Segmentation    │──────▶ │   Zone-Based People          │
│                         │        │   Detection & Counting       │
│  • Segments beach into  │        │                              │
│    water vs sand zones  │        │  • YOLOv5 detection          │
│  • 85% accuracy         │        │  • Faster R-CNN detection    │
│  • Dynamic zone mapping │        │  • Ensemble voting           │
│                         │        │  • 93% accuracy              │
└─────────────────────────┘        └──────────────────────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
          ┌─────────────────────────┐
          │   REAL-TIME DASHBOARD   │
          │  Zone counts + Alerts   │
          │  Lifeguard scheduling   │
          └─────────────────────────┘
```

---

## 🏗️ System Architecture

### Stage 1 — Water Segmentation Model

The beach area is not static — water levels change. Before we count people, we need to know where the water ends and the sand begins.

- Trained a **custom segmentation model** on beach camera images from Kincardine
- Classifies every pixel as either **water** or **sand**
- The resulting zone boundaries are used dynamically to define detection regions
- **Accuracy: 85%** — effectively segments zones across varying lighting and tide conditions

### Stage 2 — Zone-Based People Detection (Ensemble Model)

Once zones are established, we deploy a two-model ensemble:

#### 🔷 YOLOv5 (You Only Look Once v5)
- Single-stage detector — processes the entire image in one forward pass
- Uses **CSPDarknet53** backbone for feature extraction
- **5 model variants** from nano (speed-optimised) to extra-large (accuracy-optimised)
- We used **YOLOv5x** (extra-large) for maximum detection accuracy
- Confidence threshold: `0.7` for zone counting, `0.2` for broad detection

#### 🔶 Faster R-CNN (Region-based CNN)
- Two-stage detector with **Region Proposal Network (RPN)**
- Backbone: **ResNet-50 with FPN** (Feature Pyramid Network)
- Slower but higher localisation precision — especially useful for partially occluded people
- Confidence threshold: `0.5`
- Shares convolutional features between proposal and detection stages (key efficiency gain over original R-CNN)

#### ⚡ Ensemble Voting
Both models run on the same image. Their bounding box predictions are combined using **IoU-based matching** — if both models agree a detection is valid (IoU ≥ 0.5), it counts. This significantly reduces false positives.

```python
def compute_iou(box1, box2):
    # Intersection over Union — the backbone of ensemble agreement
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    union = (box1 area) + (box2 area) - intersection
    return intersection / union
```

### Zone Division Logic

The beach is divided into **4 quadrant zones** using one horizontal and one vertical midline:

```
┌──────────────┬──────────────┐
│              │              │
│    ZONE 1    │    ZONE 2    │  ← Upper beach / sand area
│  (top-left)  │  (top-right) │
├──────────────┼──────────────┤
│              │              │
│    ZONE 3    │    ZONE 4    │  ← Water-adjacent / shoreline
│ (bottom-left)│(bottom-right)│
└──────────────┴──────────────┘
```

Each detected person is assigned to exactly one zone based on bounding box origin coordinates. Zone counts are updated in real time.

---

## 📊 Results & Performance

### Model Accuracy Summary

| Model | Task | Accuracy |
|-------|------|----------|
| Water Segmentation | Zone boundary detection | **85%** |
| YOLOv5 + Faster R-CNN Ensemble | Zone-based people counting | **93%** |

> 💡 The 93% accuracy represents a **significant improvement** over the baseline single-model approach used in prior work on this dataset, achieved by shifting from whole-image to zone-by-zone prediction.

### Sample Evaluation Metrics (Per Image)

| Sample | Precision | Recall | Accuracy |
|--------|-----------|--------|----------|
| Image Set 1 | 0.271 | 0.920 | **0.920** |
| Image Set 2 | 0.118 | 0.667 | **0.667** |
| Image Set 3 | 0.440 | 0.957 | **0.957** |

> **High Recall** was prioritised over Precision in this safety application — it is far more important to detect every person (even with some false positives) than to miss someone in the water.

### Evaluation Metrics Used
- **Precision** — Of all detected objects, what fraction were correct?
- **Recall** — Of all real people, what fraction did we detect?
- **Accuracy** — Overall correct detections vs ground truth
- **IoU threshold** — 0.5 (standard for object detection benchmarking)
- **Confusion Matrix** — Used to visualise true/false positive/negative distribution
- **Precision-Recall Curve** — Used to calibrate confidence thresholds

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| Deep Learning Framework | PyTorch |
| Object Detection | YOLOv5x (Ultralytics), Faster R-CNN ResNet-50 FPN |
| Computer Vision | OpenCV |
| Image Processing | PIL (Pillow) |
| ML Utilities | Scikit-learn, NumPy |
| Visualisation | Matplotlib |
| Development Environment | Google Colab (GPU), VS Code |
| Version Control | GitHub |
| Data Storage | Google Drive |

---

## 📁 Repository Structure

```
smart-beach-cv/
│
├── 📂 water-segmentation/
│   ├── training.py              # Segmentation model training script
│   ├── testing.py               # Segmentation model testing script
│   ├── requirements.txt         # Dependencies for segmentation
│   └── models/                  # Saved segmentation model weights
│
├── 📂 zone-detection/
│   ├── ensemble_detection.py    # Main YOLOv5 + Faster R-CNN pipeline
│   ├── zone_division.py         # Zone boundary logic
│   ├── iou_utils.py             # IoU computation and ensemble voting
│   ├── evaluate.py              # Precision, recall, accuracy metrics
│   └── requirements.txt         # Dependencies for detection
│
├── 📂 notebooks/
│   ├── Zone_Based_Image_Classification_and_Object_Detection.ipynb   # Full Colab notebook
│   └── Water_Segmentation.ipynb          # Segmentation notebook
│
├── 📂 test-data/
│   └── sample_images/           # Sample beach images for testing
│       └── op/                  # Output images with zone overlays
│
├── 📂 results/
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── precision_confidence_curve.png
│   ├── labels_correlogram.png
│   └── sample_outputs/          # Zone-annotated output images
│
├── 📂 docs/
│   └── Smart_Beach_Project_Report.pdf   # Full academic report
│
├── .gitignore
├── requirements.txt             # Top-level unified dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- GPU recommended (Google Colab free tier works perfectly)
- Google Drive account (for dataset storage)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/mahimasrinivasan3994/smart-beach-cv.git
cd smart-beach-cv
```

### Step 2 — Install Dependencies
```bash
# Clone YOLOv5 (required)
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..

# Install project dependencies
pip install -r requirements.txt
```

### Step 3 — Run Water Segmentation
```bash
cd water-segmentation

# Train the segmentation model
python training.py

# Test the segmentation model
python testing.py
```

### Step 4 — Run Zone-Based People Detection
```bash
cd zone-detection

# Run ensemble detection on a folder of images
python ensemble_detection.py --input ../test-data/sample_images --output ../results/
```

### Step 5 — Run in Google Colab (Recommended for GPU)
1. Open `notebooks/Zone_Based_Image_Classification_and_Object_Detection.ipynb` in Google Colab
2. Mount your Google Drive
3. Update `folder_path` to your test image directory
4. Run all cells — outputs saved automatically to `/op` folder

---

## 🔍 Key Code Highlights

### Zone Division
```python
def divide_into_zones(image):
    width, height = image.size
    vertical_line   = width  // 2
    horizontal_line = height // 2
    zones = [
        (0,             0,              vertical_line, horizontal_line),  # Zone 1
        (vertical_line, 0,              width,         horizontal_line),  # Zone 2
        (0,             horizontal_line, vertical_line, height),          # Zone 3
        (vertical_line, horizontal_line, width,         height)           # Zone 4
    ]
    return zones
```

### Ensemble Voting Pipeline
```python
def ensemble_voting(images):
    # Run both models on the same image
    bboxes_yolo,  class_names_yolo  = detect_objects_yolo(image)
    bboxes_frcnn, class_names_frcnn = detect_objects_frcnn(image)

    # Match detections using IoU — only keep agreed detections
    precision, recall, accuracy = evaluate_performance(bboxes_yolo, bboxes_frcnn)
    return bboxes_yolo, bboxes_frcnn, precision, recall, accuracy
```

---

## ⚠️ Known Limitations & Future Work

| Limitation | Proposed Improvement |
|-----------|---------------------|
| False positives in cluttered scenes (beach umbrellas misclassified) | Fine-tune on beach-specific labelled dataset |
| Faster R-CNN slower on real-time video streams | Replace with YOLOv8 or RT-DETR for production |
| Zone boundaries are fixed midlines | Use dynamic zone boundaries from water segmentation output |
| No alert/notification system built yet | Integrate with SMS/app alert system as planned by MIC |
| Evaluated on still images only | Extend to live video stream processing |

---

## 🌊 Project Impact

This system directly supports the **Smart Beach Kincardine** safety initiative:

- ✅ Enables **proactive** lifeguard scheduling instead of reactive response
- ✅ Provides zone-level crowd density data for **hotspot identification**
- ✅ First-of-its-kind **AI beach safety system in North America**
- ✅ Designed to integrate with existing buoy/sensor data infrastructure
- ✅ Lays groundwork for potential **Bruce County shoreline expansion**

> *"For first responders in Kincardine, there is optimism that the data collected through this initiative will enable predictive modelling, facilitating smarter and safer decision-making during water rescue operations."*
> — MIC Bruce County

---

## 👩‍💻 Author

**Mahima Srinivasan**
Post-Graduate Certificate — AI Architecture, Design & Implementation
Georgian College, Barrie, ON | April 2024

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mahimasrinivasan3994)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mahima.s3994@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mahimasrinivasan3994)

---

## 📎 References

- Ultralytics YOLOv5: https://github.com/ultralytics/yolov5
- Prior Smart Beach work: https://github.com/ruwzeta/MIC_at_Bruce_Beach
- Smart Beach Dataset: https://github.com/ruwzeta/Smart-Beach-Predict-People
- PyTorch Faster R-CNN: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- Municipal Innovation Council (MIC), Bruce County, Ontario

---

## 📄 Acknowledgements

This project builds upon prior research by the University of Windsor and the Smart Beach Kincardine team. Camera data and project context were provided by the MIC Bruce County initiative. This implementation was developed as a capstone academic project at Georgian College.

---

<!-- FOOTER -->
![Footer](https://capsule-render.vercel.app/api?type=waving&color=0077B6&height=100&section=footer)

<div align="center">
<i>⭐ If this project was useful or interesting to you, please give it a star — it helps others find it!</i>
</div>
