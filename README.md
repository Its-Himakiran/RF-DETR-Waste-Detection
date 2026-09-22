# RF-DETR: Multi-Class Waste Detection Using a Transformer Framework Based on DINOv2 with Real-World Field Validation

**DR. RVR NRI INSTITUTE OF TECHNOLOGY DEEMED TO BE UNIVERSITY**
*Department of Computer Science and Engineering*

A deep learning vision transformer system for multi-class environmental waste detection using **RF-DETR-Medium** powered by a self-supervised **DINOv2** backbone.

The project covers end-to-end model training, empirical held-out benchmark evaluations across two distinct taxonomies (10-class and 24-class), real-world GPS-tagged field validation at an uncontrolled disposal site in Tallamudi, Andhra Pradesh, and an interactive Streamlit web dashboard for real-time inference.

---

## 📌 Project Overview

Solid Waste Management (SWM) is one of the most critical environmental challenges facing rapidly urbanizing regions. Traditional automated sorting pipelines rely heavily on convolutional detectors (such as YOLO variants and Faster R-CNN), which often experience performance degradation as taxonomies scale beyond 10 categories or when dealing with visual clutter and deformed objects.

This project implements **RF-DETR (Real-Time Detection Transformer)**, integrating self-supervised visual representations from **DINOv2** for multi-class waste localization and classification without hand-designed components like anchor generators or Non-Maximum Suppression (NMS).

### Main Components

* **RF-DETR-Medium Architecture:** End-to-end transformer detector leveraging a frozen/fine-tuned self-supervised DINOv2 backbone.
* **Dual Waste Taxonomies:**

  * **TACO v3:** 10 coarse/fine litter categories (class-imbalanced open-environment benchmark).
  * **MultipleWaste v3:** 24 fine-grained recyclable and non-recyclable material classes.
* **Cloud-Scale Model Training:** Independent fine-tuning protocols executed on Kaggle dual-NVIDIA T4 GPUs.
* **Rigorous Empirical Evaluation:** Evaluated via COCO mAP@50, mAP@50:95, Precision, Recall, F1-score, and per-class AP breakdown.
* **Real-World Uncontrolled Field Trial:** On-site validation using GPS-tagged smartphone imagery collected at a rural disposal site in Tallamudi, Andhra Pradesh, India.
* **Interactive Streamlit Application:** Browser-based deployment supporting dual checkpoint selection, real-time bounding box rendering, confidence filtering, and inference latency telemetry.

---

## 🎯 Objectives

* Overcome CNN capacity limitations in complex multi-class waste scenarios (>10 classes).
* Benchmark transferability of self-supervised DINOv2 features for solid waste detection.
* Train and evaluate RF-DETR across both 10-class (TACO v3) and 24-class (MultipleWaste v3) corpora.
* Conduct field validation on uncontrolled, outdoor, geo-referenced waste imagery.
* Provide an accessible, production-ready Streamlit web interface for municipal monitoring.

---

## 🧠 Model Architecture

RF-DETR utilizes a hybrid transformer architecture designed for real-time throughput while maintaining global contextual attention.

```text
Input Waste Image (704 × 704)
          │
          ▼
DINOv2 Self-Supervised Vision Transformer Backbone (ViT-Medium)
          │
          ▼
Multi-Scale Feature Projector & Lightweight Deformable Encoder
          │
          ▼
Transformer Decoder (Object Queries + Cross-Attention)
          │
          ▼
Bipartite Hungarian Matching (End-to-End, No NMS)
          │
          ▼
Detected Waste Objects
      ├── Class Label (10-Class TACO / 24-Class MultipleWaste)
      ├── Bounding Box [x_min, y_min, x_max, y_max]
      └── Detection Confidence Score
```

---

## 🗂️ Datasets & Taxonomies

The framework evaluates detection resilience across two distinct benchmark corpora.

### 1. TACO v3 — 10 Classes

**Categories:**

* Bottle
* Bottle cap
* Can
* Cigarette
* Cup
* Lid
* Other
* Plastic bag and wrapper
* Pop tab
* Straw

**Dataset Characteristics:**

* Training images: **3,561**
* Training annotations: **11,462**
* Validation images: **150**
* Test images: **149**
* Test annotations: **443**

**Source:** TACO YOLO 10-Class Dataset on Roboflow Universe

---

### 2. MultipleWaste v3 — 24 Classes

**Categories:**

* Cardboard
* Carton packaging
* Cigarette
* Clean paper
* Clear plastic
* Contaminated paper
* Food packaging
* Food scraps
* Glass
* Medical waste
* Metal
* Paper bag
* Paper cup
* Plastic bottle
* Plastic container
* Plastic cup
* Plastic lid
* Plastic packaging
* Plastic utensil
* Printed cardboard
* Sanitary waste
* Straw
* Styrofoam
* Wood

**Dataset Characteristics:**

* Training images: **2,199**
* Training annotations: **9,737**
* Validation images: **18**
* Test images: **48**
* Test annotations: **293**

**Source:** Multiple Waste Dataset on Roboflow Universe

---

## ☁️ Kaggle Training, Checkpoints & Artifacts

All models were trained using **PyTorch** and **PyTorch Lightning** on **2× NVIDIA Tesla T4 GPUs** with **bf16-mixed precision**.

### Kaggle Training Notebooks

* **MultipleWaste v3 (24 Classes):** View Kaggle Training Notebook
* **TACO v3 (10 Classes):** View Kaggle Training Notebook

### Trained Model Weights

Due to GitHub file size limits (>100 MB), model checkpoints (`.pth`) are not committed directly to this repository.

You can obtain them directly from the **Output** section of each Kaggle notebook:

```text
multiwaste_rfdetr.pth
```

Source:

```text
rfdetr_out/checkpoint_best_regular.pth
```

from the MultipleWaste notebook.

For TACO:

```text
taco_rfdetr.pth
```

Source:

```text
rfdetr_out/checkpoint_best_regular.pth
```

from the TACO notebook.

Place both `.pth` files inside the `models/` directory prior to running the local application.

---

## 📊 Evaluation & Empirical Results

### Quantitative Benchmark Comparison

| Dataset          | Detector       | Classes | Input Size | mAP@50 (%) | mAP@50:95 (%) | Test Recall (%) |
| ---------------- | -------------- | ------: | ---------- | ---------: | ------------: | --------------: |
| TACO v3          | RF-DETR-Medium |      10 | 704 × 704  |      52.5% |         41.9% |           68.2% |
| MultipleWaste v3 | RF-DETR-Medium |      24 | 704 × 704  |      79.4% |         69.9% |           84.4% |

### Confusion Matrix Performance

**MultipleWaste v3 @ Confidence > 0.25**

* **Overall Precision:** 78.6%
* **Overall Recall:** 84.4%
* **Overall F1-Score:** 81.4%
* **Correct Matches:** 228 of 293 ground-truth targets (diagonal accuracy)

Complete training loss trajectories, PR curves, and per-class confusion matrices are available in the `Results/` directory.

---

## 🚀 Application & Key Features

### Dual-Model Switching

Toggle between the **TACO (10-class)** and **MultipleWaste (24-class)** models on the fly.

### Multi-Object Localization

Detects multiple overlapping litter items within a single high-resolution image.

### Real-Time Confidence Filtering

Dynamic threshold slider from **0.00 – 1.00** to eliminate low-confidence false positives.

### Per-Class Category Counts

Summarizes detected material types and overall category distribution.

### Inference Speed Monitoring

Displays backend vision transformer latency in milliseconds.

### Image Export

Download annotated images with labeled bounding boxes directly from the browser.

---

## 📂 Project Structure

```text
RF-DETR-Waste-Detection/
│
├── core/
│   ├── __init__.py
│   └── detector.py
│
├── models/
│   ├── multiwaste_rfdetr.pth
│   └── taco_rfdetr.pth
│
├── Results/
│   ├── multiwaste_v3-MEDIUM 704 PX/
│   │   ├── confusion_matrix.png
│   │   ├── F1_curve.png
│   │   └── PR_curve.png
│   │
│   └── taco-rfdetr-v3-MEDIUM 704 PX/
│       ├── confusion_matrix.png
│       ├── F1_curve.png
│       └── PR_curve.png
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File / Directory   | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `core/`            | Core inference functionality                              |
| `core/detector.py` | Inference pipeline and model wrapper                      |
| `models/`          | Trained RF-DETR model checkpoints                         |
| `Results/`         | Evaluation plots and performance results                  |
| `app.py`           | Interactive Streamlit web interface                       |
| `requirements.txt` | Project dependencies                                      |
| `.gitignore`       | Git exclusions such as `.venv`, `.pth`, and `__pycache__` |
| `README.md`        | System documentation                                      |

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Make sure the following are installed:

* Python **3.10 or higher**
* Git
* NVIDIA GPU with CUDA support *(recommended for low-latency inference; CPU is also supported)*

### Verify Your Environment

```bash
python --version
git --version
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/Its-Himakiran/RF-DETR-Waste-Detection.git
cd RF-DETR-Waste-Detection
```

---

### 3. Create & Activate Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt (CMD)

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 4. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

### 5. Download Model Checkpoints

Download the trained weights from the Kaggle notebooks and place them inside the `models/` directory.

The directory should contain:

```text
models/
├── multiwaste_rfdetr.pth
└── taco_rfdetr.pth
```

---

## 🖥️ Running the Application

Ensure your virtual environment is active, then run:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

```text
Local URL: http://localhost:8501
```

---

## 🎯 How to Use

### Step 1 — Launch the Dashboard

Run:

```bash
streamlit run app.py
```

### Step 2 — Select Model Checkpoint

In the sidebar, choose between:

* **MultipleWaste v3 (24 Classes)**
* **TACO v3 (10 Classes)**

### Step 3 — Set Confidence Threshold

Adjust the confidence threshold slider.

Recommended range:

```text
0.25 – 0.40
```

### Step 4 — Upload Image

Upload any:

* `.jpg`
* `.jpeg`
* `.png`

outdoor scene or litter photograph.

### Step 5 — Inspect Detections

Review:

* Predicted bounding boxes
* Category labels
* Object summary counts
* Inference latency

### Step 6 — Download Output

Click **Download Result** to save the labeled image.

---

## 🔧 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'rfdetr'`

Ensure the virtual environment is active and install RF-DETR:

```bash
pip install "rfdetr[train,loggers]"
```

---

### Error: `FileNotFoundError: models/multiwaste_rfdetr.pth`

Verify that the `.pth` files are downloaded from Kaggle and placed directly inside the `models/` directory.

The filenames must exactly match:

```text
multiwaste_rfdetr.pth
taco_rfdetr.pth
```

---

### Error: Port 8501 Is Already in Use

Launch Streamlit on a secondary port:

```bash
streamlit run app.py --server.port 8502
```

---

## 🧰 Technology Stack

### Core Framework

* Python 3.10+
* PyTorch
* Torchvision

### Model Family

* RF-DETR (Real-Time Detection Transformer)
* DINOv2 (Meta AI)

### Application & Deployment

* Streamlit
* Supervision
* OpenCV
* Pillow

### Hardware & Cloud

* Kaggle
* 2× NVIDIA Tesla T4 GPUs
* Roboflow Universe

---

## 📚 Project Resources

### GitHub Repository

**Its-Himakiran/RF-DETR-Waste-Detection**

### Kaggle Notebooks

* **MultipleWaste:** `multiplewaste-rfdetr-wastedetection`
* **TACO:** `taco-rfdetr-v3-retrain`

### Roboflow Datasets

* **TACO:** `taco-yolo-10-class-original` (v4)
* **MultipleWaste:** `multiple-waste-dataset` (v8)

---

## 👨‍💻 Authors

**Mustina Hima Kiran**
**Nagaraju Shyam Vara Prasad Raju**
**Karnikula Mourya Mahesh**
**Mohammad Amman Fawaz**
**Lakshmi Raj Ravi**
**Dr. K. V. Sambasiva Rao** — Director, Research & Development

**Department of Computer Science and Engineering**
**DR. RVR NRI INSTITUTE OF TECHNOLOGY DEEMED TO BE UNIVERSITY**

**Academic Period:** 2023–2027

---

## 📄 License

This repository is distributed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for further details.
