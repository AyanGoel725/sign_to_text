# Sign Language to Text

A real-time American Sign Language (ASL) recognition system that translates hand gestures into text using computer vision and machine learning.

## Overview

This project uses **MediaPipe Hands** for landmark detection and a **Keras neural network** to classify ASL gestures. It supports the full alphabet (A-Z) plus special gestures for space and delete, enabling users to spell out words and sentences through sign language.

## Features

- **Real-time gesture recognition** via webcam
- **26 letter alphabet** (A-Z) recognition
- **Special gestures**: `space` and `del` (delete)
- **Sentence building** with majority-vote prediction smoothing
- **Runtime model switching** — hot-swap between models with keyboard shortcuts
- **Adjustable smoothing** — tune the prediction buffer size live
- **Auto-punctuation** after pauses
- **Data collection tool** for training custom gestures

## Project Structure

```
sign_to_text/
├── model/                          # Trained model artifacts
│   ├── test_model.h5               # Original model (96.78% accuracy, has leakage)
│   ├── test_model_grouped.h5       # Jump-based grouped model (96.60%, has leakage)
│   ├── test_model_clustered.h5     # Cluster-based model (96.26%, no leakage)
│   └── label_encoder.pkl           # Scikit-learn label encoder
├── ml/                             # Training & evaluation pipeline
│   ├── collect.py                  # Data collection tool - capture hand landmarks
│   ├── training.py                 # Train the MLP classifier (random split)
│   ├── train_grouped.py            # Train with jump-based session grouping
│   ├── train_clustered.py          # Train with similarity-based clustering (recommended)
│   └── eval.py                     # Model evaluation with leakage analysis
├── api/                            # FastAPI backend
│   └── main.py                     # REST API for inference
├── static/                         # Frontend web app
│   ├── index.html                  # Web UI
│   ├── app.css                     # Web UI styling
│   └── app.js                      # Frontend JS with MediaPipe
├── tests/                          # Tests
│   └── test_api.py                 # API tests
├── app.py                          # Real-time inference with sentence building (offline mode)
├── test_webcam.py                  # Interactive model comparison tool
├── data.csv                        # Training dataset (57k samples from Kaggle)
├── cluster_statistics.csv          # Per-class cluster distribution
├── Dockerfile                      # Container setup (planned)
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

### Prerequisites

- Python 3.12 (recommended) or 3.10-3.12
- Webcam
- Windows/macOS/Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/AyanGoel725/sign_to_text.git
cd sign_to_text
```

2. **Create a virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Run the Web App (Recommended)

**FastAPI + Web Frontend** (real-time inference in browser):

1. Start the API server:
```bash
uvicorn api.main:app --reload
```

2. Open http://localhost:8000/ in your browser

3. Click "Start Camera" and begin signing

**Features:**
- MediaPipe Hands integration for landmark detection
- Real-time predictions with confidence filtering (min 85%)
- Majority-vote smoothing (5-frame buffer, 80% threshold)
- Sentence building with `space` and `del` gestures
- Auto-punctuation after 3s of no hand detection
- Manual backspace and clear controls

**Tuning parameters** (in `static/app.js`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `SMOOTHING_WINDOW` | 5 | Buffer size (frames) |
| `SMOOTHING_THRESHOLD` | 0.80 | Majority % required (4/5) |
| `MIN_CONFIDENCE` | 0.85 | Min confidence to enter buffer |
| `PAUSE_BETWEEN_SIGNS` | 0.5s | Repeat-letter delay |

Open the browser console to see confidence values for all predictions, including filtered ones.

### Run Offline Recognition

**Desktop app** (OpenCV-based sentence builder):
```bash
python app.py
```
- Builds sentences letter-by-letter
- Shows running text on screen
- Supports `space` and `del` gestures
- Auto-adds period (`.`) after 3 seconds of no hand detection

**Controls:**
| Key | Action |
|-----|--------|
| `1`-`9` | Switch between loaded models |
| `+` / `=` | Increase smoothing window N |
| `-` | Decrease smoothing window N |
| `ESC` | Exit |

**Prediction smoothing:** The desktop app uses majority-vote smoothing — it keeps a rolling buffer of the last N predictions (default N=10) and only commits a letter when one prediction holds ≥70% of the buffer. Raw per-frame predictions and committed letters are printed to the console so you can see the difference.

### Run API Tests

```bash
pytest tests/
```

Tests the FastAPI endpoints (`/health` and `/predict` with valid/invalid inputs) using FastAPIs `TestClient`.

### Collect Training Data

To train on your own gestures or add new signs:

```bash
python ml/collect.py
```

1. Position your hand in view of the webcam
2. Press **A-Z** on your keyboard to label and save the current gesture
3. Collect 50-100 samples per letter for best results
4. Data saved to `sign_data.csv`
5. Press **Q** to quit

### Train Your Own Model

After collecting data:

```bash
# Basic training (random split — may have data leakage)
python ml/training.py

# Recommended: cluster-based training (no leakage)
python ml/train_clustered.py

# Alternative: jump-based session grouping
python ml/train_grouped.py
```

`training.py` will:
- Load training data from `data.csv`
- Train a 3-layer MLP neural network with early stopping
- Save the model as `model/test_model.h5`
- Save the label encoder as `model/label_encoder.pkl`

### Evaluate the Model

To evaluate the current model against `data.csv`:

```bash
python ml/eval.py
```

Generates:
- Overall accuracy comparison (random vs positional split)
- Near-duplicate leakage detection (tests for ~40% data leakage)
- Session boundary detection via consecutive jumps
- Per-class accuracy for weakest classes
- Detailed classification report
- Confusion matrix

### Test Models Interactively

To compare all three models with your webcam:

```bash
python test_webcam.py
```

Features:
- Select which model to load at startup
- Switch between models live (press **M**)
- Shows prediction confidence percentage
- Press **SPACE** to clear sentence
- Press **ESC** to exit

## Model Architecture

**Input:** 63 features (21 hand landmarks × 3 coordinates: x, y, z)

**Architecture:**
```
Dense(128, relu) 
→ Dropout(0.2) 
→ Dense(64, relu) 
→ Dropout(0.2) 
→ Dense(28, softmax)  # 26 letters + space + del
```

**Training:**
- Optimizer: Adam
- Loss: Categorical crossentropy
- 80/20 train/test split
- 10 epochs, batch size 128

**Performance:** 
- **Original model (random split):** 96.78% accuracy, but **39.52% of test samples have near-duplicates in training** due to data leakage
- **Grouped model (jump-based):** 96.60% accuracy, but **39.34% leakage** remains
- **Clustered model (similarity-based, recommended):** **96.26% accuracy with 0.00% leakage** — true generalization performance

The default `test_model.h5` was trained on the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) (~57k samples) but suffers from data leakage. **Use `test_model_clustered.h5` for production** — it was trained with proper group-aware splitting where visually similar samples (distance < 0.05) are kept entirely within train or test, never split across both.

## How It Works

1. **Hand Detection:** MediaPipe Hands detects hand landmarks (21 points per hand)
2. **Feature Extraction:** Extract x, y, z coordinates → 63-dimensional vector
3. **Classification:** Neural network predicts gesture class
4. **Smoothing:** Majority-vote over rolling buffer (last N frames, default N=10, 70% threshold) confirms stable gestures before committing
5. **Output:** Display predicted letter or build sentence

## Known Issues

1. **Similar gestures:** The model can sometimes confuse visually similar gestures like `M` and `N` (M is three fingers over the thumb, N is two fingers). This mirrors real-world ASL logic but requires precise hand positioning.
2. **CSV filename mismatch for collection:** `ml/collect.py` writes to `sign_data.csv`, but `ml/training.py` reads `data.csv`. Rename the file before training custom data.

## Requirements

- `opencv-python` 4.9.0+ (for video capture and display)
- `mediapipe` 0.10.18 (for hand landmark detection)
- `tensorflow` 2.17.0+ (for neural network)
- `scikit-learn` 1.5.0+ (for label encoding)
- `pandas` 2.2.0+ (for data handling)
- `numpy` 1.23.0-1.26.x (mediapipe requires <2.0)
- `matplotlib` 3.9.0+ (for visualization)

See `requirements.txt` for full dependency list.

## Tips for Best Results

- **Lighting:** Use consistent, bright lighting
- **Background:** Plain background improves hand detection
- **Hand position:** Keep hand centered and at comfortable distance
- **Training data:** Collect 50-100 samples per gesture in varied positions
- **Gesture stability:** Hold each sign steady for ~1 second for recognition

## Troubleshooting

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"Webcam not found"**
- Check camera permissions
- Try changing camera index in `cv2.VideoCapture(0)` to `(1)` or `(2)`

**"Model predictions are wrong"**
- Retrain with more diverse data
- Check feature ordering bug (see Known Issues #1)
- Ensure consistent lighting between training and inference

**"numpy version conflict"**
- MediaPipe requires numpy<2.0, install: `pip install "numpy<2.0"`

## Future Improvements

- [x] FastAPI backend for web-based inference
- [x] Web frontend with webcam capture
- [ ] Add word-level recognition
- [ ] Support dynamic gestures (motion-based signs)
- [ ] Add grammar/autocorrect
- [ ] Mobile app deployment
- [ ] Docker containerization
- [ ] Support for other sign languages

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [TensorFlow/Keras](https://www.tensorflow.org/) for deep learning framework
- ASL community for gesture references

## Contact

**Ayan Goel**  
GitHub: [@AyanGoel725](https://github.com/AyanGoel725)  
Project Link: [https://github.com/AyanGoel725/sign_to_text](https://github.com/AyanGoel725/sign_to_text)

---

Built with ❤️ for accessibility
