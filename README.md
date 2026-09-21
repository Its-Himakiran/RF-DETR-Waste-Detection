# RF-DETR: Multi-Class Waste Detection Using a Transformer Framework Based on DINOv2 with Real-World Field Validation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Backbone DINOv2](https://img.shields.io/badge/Backbone-DINOv2--Medium-success.svg)](https://github.com/facebookresearch/dinov2)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end vision transformer pipeline for solid waste localization and multi-class categorization. The framework uses **RF-DETR-Medium (Receptive-Field Detection Transformer)** with a self-supervised **DINOv2-Medium** backbone and supports real-world field image inference through an interactive Streamlit application.

The models are trained and evaluated independently on **TACO v3 (10 classes)** and **MultipleWaste v3 (24 classes)** without merging their class taxonomies.

---

## 📊 Benchmark & Validation Results

All evaluations were conducted on held-out validation sets using an operating input resolution of **704 × 704 px**.

| Study | Detector | Dataset | Classes | mAP@50 | mAP@50:95 |
|:---|:---|:---|---:|---:|---:|
| **This Work** | **RF-DETR-Medium** | **TACO v3** | **10** | **52.5** | **41.9** |
| **This Work** | **RF-DETR-Medium** | **MultipleWaste v3** | **24** | **79.4** | **69.9** |

---

## 🛠️ Training Configuration

Both models were fine-tuned independently using the following configuration:

| Hyperparameter | Value |
|:---|:---|
| **Model** | RF-DETR-Medium |
| **Framework** | PyTorch |
| **Backbone** | DINOv2-Medium (self-supervised ViT) |
| **Optimizer** | AdamW |
| **Learning Rate** | 1 × 10⁻⁴ |
| **Scheduler** | Cosine |
| **Warm-up** | 2 epochs |
| **Weight Decay** | 1 × 10⁻⁴ |
| **Precision** | BF16 Mixed Precision |
| **GPU** | 2 × NVIDIA T4 (Kaggle) |
| **Input Resolution** | 704 × 704 px |
| **Checkpoint Selection** | Best EMA validation mAP@50:95 |

---

## 🚀 Application & Key Features

- **Transformer Architecture:** End-to-end RF-DETR detection with bipartite matching.
- **On-Demand Inference:** Detection runs when the user clicks **`⚡ Detect Waste`**.
- **Both (Comparison):** Side-by-side inference using TACO v3 and MultipleWaste v3 models.
- **TACO RF-DETR:** Detection using the 10-class TACO v3 model.
- **MultipleWaste RF-DETR:** Detection using the 24-class MultipleWaste v3 model.
- **Real-Time Analytics:** Object counts, confidence scores, and inference latency.
- **Confidence Filtering:** Adjustable threshold from **0.05 to 1.00**.
- **Batch Processing:** Supports multiple images and ZIP archives.
- **Supported Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`.
- **Real-World Inference:** Supports detection on user-uploaded field images.

---

## 📂 Project Structure

```text
WasteDetectionApp/
│
├── .streamlit/
│   └── config.toml
│
├── core/
│   ├── __init__.py
│   └── detector.py
│
├── models/
│   ├── taco_rfdetr.pth
│   ├── multiwaste_rfdetr.pth
│   └── README.md
│
├── Results/
│   ├── taco-rfdetr-v3-MEDIUM 704PX/
│   └── multiwaste_v3-MEDIUM 704 PX/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation & Setup

Follow these steps to run the project on Windows, macOS, or Linux.

## 1. Prerequisites

Install:

- **Python 3.10 or higher**
- **Git**
- Internet connection for package installation

Check Python:

```bash
python --version
```

Check Git:

```bash
git --version
```

### Windows

During Python installation, enable:

```text
Add Python to PATH
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/<YOUR-GITHUB-USERNAME>/RF-DETR-Waste-Detection.git
cd RF-DETR-Waste-Detection
```

Replace `<YOUR-GITHUB-USERNAME>` with your GitHub username.

Alternatively, download the repository as a ZIP, extract it, and open a terminal inside the project folder.

---

## 3. Create & Activate Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, the terminal should show:

```text
(.venv)
```

---

## 4. Install Dependencies

With `(.venv)` activated:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install rfdetr supervision
```

Verify RF-DETR:

```bash
pip show rfdetr
```

---

## 5. Add Model Checkpoints

Place the trained model weights inside the `models/` directory:

```text
models/
├── taco_rfdetr.pth
└── multiwaste_rfdetr.pth
```

The filenames must match **exactly**:

```text
taco_rfdetr.pth
multiwaste_rfdetr.pth
```

These large `.pth` files may be distributed separately through GitHub Releases, Git LFS, or cloud storage.

---

# 🖥️ Running the Application

Make sure the terminal is inside the project directory and `(.venv)` is active.

Run:

```bash
streamlit run app.py
```

Streamlit will display:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open:

```text
http://localhost:8501
```

If the browser does not open automatically, copy the local URL into Chrome, Edge, or Firefox.

---

# 🎯 How to Use

### 1. Select Detection Mode

Choose one:

```text
Both (Comparison)
TACO RF-DETR
MultipleWaste RF-DETR
```

### 2. Set Confidence Threshold

Adjust the confidence threshold:

```text
Minimum: 0.05
Maximum: 1.00
Default: 0.50
```

### 3. Upload Images

Upload:

```text
.jpg
.jpeg
.png
.bmp
```

You can also upload a `.zip` file containing multiple images.

### 4. Run Detection

Click:

```text
⚡ Detect Waste
```

### 5. View Results

The application displays:

- Bounding boxes
- Waste class labels
- Confidence scores
- Per-class object counts
- Total detections
- Mean prediction confidence
- Inference latency in milliseconds

---

# 🔧 Troubleshooting

### `No module named 'rfdetr'`

Install RF-DETR inside the active virtual environment:

```bash
pip install rfdetr
```

---

### `No module named 'supervision'`

Run:

```bash
pip install supervision
```

---

### `File does not exist: app.py`

Make sure the terminal is inside the project directory containing `app.py`:

```bash
cd WasteDetectionApp
```

Then run:

```bash
streamlit run app.py
```

---

### `running scripts is disabled on this system`

In Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

---

### `TracerWarning: Converting a tensor...`

This is a PyTorch runtime warning that may occur during model tracing. If the application starts and inference works correctly, it is not a fatal error.

---

### `args.num_queries absent; inferred ckpt_num_queries=300`

RF-DETR may display:

```text
args.num_queries absent; inferred ckpt_num_queries=300
```

If the model loads successfully and the application runs normally, this is a configuration warning rather than a fatal error.

---

### `Checkpoint not found`

Verify that both trained checkpoints exist:

```text
models/
├── taco_rfdetr.pth
└── multiwaste_rfdetr.pth
```

Make sure the filenames are spelled exactly as shown.

---

### Port 8501 Already in Use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

---

# 📈 Results Directory

Evaluation outputs are stored under:

```text
Results/
├── taco-rfdetr-v3-MEDIUM 704PX/
└── multiwaste_v3-MEDIUM 704 PX/
```

These directories may contain:

- Precision-recall curves
- Confusion matrices
- Evaluation results
- Training/evaluation logs

---

# 🧰 Technology Stack

**Python · PyTorch · RF-DETR · DINOv2 · Supervision · Streamlit · OpenCV · NumPy · Kaggle**

---

# 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file for the complete license text.