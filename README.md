# Sign-to-Text: ASL Hand-Gesture Classifier

> Real-time ASL fingerspelling recognition (A-Z, space, del) with a focus on rigorous data splitting and low-latency inference.

[![CI](https://github.com/AyanGoel725/sign_to_text/actions/workflows/ci.yml/badge.svg)](https://github.com/AyanGoel725/sign_to_text/actions/workflows/ci.yml)  
**[Live Demo](https://signtotext-6xhl.onrender.com/)** *(Note: Free-tier hosting may have a 30-60s cold start delay)*

## Overview

This project is a real-time American Sign Language (ASL) fingerspelling recognizer. 
It captures webcam frames, extracts 3D hand landmarks using MediaPipe, and passes those coordinates through a Keras Multi-Layer Perceptron (MLP) classifier to predict the corresponding letter.

## Live Demo & Architecture

The application is hosted as a real-time browser demo. For both privacy and efficiency, **no video frames are sent to the server.**

Instead, the client uses MediaPipe JS to extract the 3D hand landmarks directly in the browser. Only the lightweight numerical coordinates (a flattened vector of 21 landmarks) are sent to the backend via HTTP POST, ensuring fast inference and low bandwidth usage.

### Architecture Data Flow

```text
┌─────────────────┐                                  ┌──────────────────────┐
│  Browser Client │                                  │    FastAPI Server    │
│                 │                                  │                      │
│ 1. Webcam Frame │                                  │ 3. /predict endpoint │
│ 2. MediaPipe JS │───(JSON: 21 (x,y,z) Landmarks)──▶│ 4. Keras MLP Model   │
│                 │                                  │                      │
│ 7. UI Display   │◀──(JSON: Class & Confidence)─────│ 5. Classification    │
│ 6. Smoothing    │                                  └──────────────────────┘
└─────────────────┘
```

## 🔍 The Data Leakage Investigation

The most critical engineering work in this project went into ensuring honest evaluation metrics. A common pitfall in computer vision datasets is data leakage between train and test sets, which artificially inflates model accuracy. 

1. **Initial findings**: The project began with self-collected data, which yielded ~97% accuracy on a random train/test split. However, this was misleading. The data was inherently grouped by recording session (capturing many frames of a single sustained sign). A random shuffle leaked near-duplicate frames from the same continuous motion into both the train and test sets.
2. **Kaggle Dataset Audit**: We transitioned to a [57k-sample Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), but explicitly tested it for the same flaw.
3. **Proving Leakage**: By running a nearest-neighbor distance check between the train and test sets, we discovered **39.5% of the test samples were near-duplicates** of training samples.
4. **The Fix**: We implemented DBSCAN-based session clustering (grouping near-identical frames using a distance threshold of 0.05) to reconstruct the original recording sessions. We then used `GroupShuffleSplit` on these generated clusters to guarantee that no session spanned both splits.

The result is a verified, leakage-free evaluation that remains highly accurate:

| Split method | Accuracy | Leakage |
|---|---|---|
| Random split | 96.78% | 39.5% |
| DBSCAN cluster-based split | **96.26%** | **0%** |

## Model Performance & Ambiguity

On the honest (zero-leakage) test set, the overall accuracy is 96.26%. Performance is generally excellent, but certain specific classes exhibit lower recall:
* **M** (~90%)
* **N** (~79%)
* **T** (~93%)
* **R** (~93%)
* **S** (~94%)

This drop is not an artifact of data imbalance or leakage—these exact classes were fundamentally weak across both our leaky and strict splits. This represents **genuine handshape similarity**. In ASL, letters like M, N, T, and S vary primarily by the position of the thumb tucked under varying fingers, an occlusion that is notoriously difficult for single-camera 2D/3D landmark estimation to resolve perfectly.

## Real-time Inference Tuning

Converting a static frame classifier to a real-time temporal system introduces flicker. To solve this, predictions pass through a majority-vote smoothing buffer.

Initially, a large 10-frame window was required to stabilize the output, resulting in ~660ms of perceived latency. 
By introducing **confidence filtering**, predictions below an 0.85 confidence threshold are discarded entirely. Filtering the noise *before* the buffer allowed us to shrink the smoothing window down to 5 frames, cutting prediction latency in half to **~300ms** while maintaining stability.

## Tech Stack

* **Machine Learning**: TensorFlow/Keras, scikit-learn
* **Computer Vision**: MediaPipe Hands (JavaScript + Python)
* **API Backend**: FastAPI, Uvicorn
* **DevOps**: Docker, GitHub Actions (CI)

## Project Structure

```text
.
├── api/          # FastAPI application routes and server config
├── ml/           # Data preprocessing, training scripts, and DBSCAN logic
├── model/        # Serialized Keras artifacts (.h5)
├── static/       # Frontend (HTML, CSS, JS with MediaPipe integration)
├── tests/        # Pytest test suite
└── Dockerfile    # Containerization config for deployment
```

## Running Locally

1. Clone the repository and navigate to the project root:
   ```bash
   git clone https://github.com/AyanGoel725/sign_to_text.git
   cd sign_to_text
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the FastAPI development server:
   ```bash
   uvicorn api.main:app --reload
   ```
4. Open your browser to `http://localhost:8000` to view the UI and grant camera permissions.

## Running with Docker

To build and run the application in a container:
```bash
docker build -t sign-to-text .
docker run -p 8000:8000 sign-to-text
```
The app will be available at `http://localhost:8000`.

## Running Tests

To run the automated test suite:
```bash
pytest tests/
```

## Collect Your Own Training Data

To train on your own gestures or add new signs:

```bash
python ml/collect.py
```

**Workflow:**
1. Position your hand in view of the webcam
2. Press **A-Z** on your keyboard to label and save the current gesture
3. Collect **50-100 samples per letter** for best results
4. Data is saved to `sign_data.csv`
5. Press **Q** to quit

**Note**: `ml/collect.py` writes to `sign_data.csv`, but `ml/training.py` reads `data.csv`. Rename the file before training on custom data.

## Train Your Own Model

After collecting data (or downloading the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)):

```bash
# Basic training (random split — may have data leakage)
python ml/training.py

# Recommended: cluster-based training (no leakage)
python ml/train_clustered.py

# Alternative: jump-based session grouping
python ml/train_grouped.py
```

**What `training.py` does:**
- Loads training data from `data.csv`
- Trains a 3-layer MLP neural network with early stopping
- Saves the model as `model/test_model.h5`
- Saves the label encoder as `model/label_encoder.pkl`

## Evaluate the Model

To evaluate the current model against `data.csv`:

```bash
python ml/eval.py
```

**Generates:**
- Overall accuracy comparison (random vs positional split)
- Near-duplicate leakage detection (tests for ~40% data leakage)
- Session boundary detection via consecutive jumps
- Per-class accuracy for weakest classes
- Detailed classification report
- Confusion matrix

## Test Models Interactively

To compare all three models with your webcam:

```bash
python test_webcam.py
```

**Features:**
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
- Original model (random split): 96.78% accuracy, but **39.52% of test samples have near-duplicates in training** due to data leakage
- Grouped model (jump-based): 96.60% accuracy, but **39.34% leakage remains**
- Clustered model (similarity-based, **recommended**): **96.26% accuracy with 0.00% leakage** — true generalization performance

The default `test_model.h5` was trained on the Kaggle ASL Alphabet Dataset (~57k samples) but suffers from data leakage. Use `test_model_clustered.h5` for production — it was trained with proper group-aware splitting where visually similar samples (distance < 0.05) are kept entirely within train or test, never split across both.

## How It Works

1. **Hand Detection**: MediaPipe Hands detects hand landmarks (21 points per hand)
2. **Feature Extraction**: Extract x, y, z coordinates → 63-dimensional vector
3. **Classification**: Neural network predicts gesture class
4. **Smoothing**: Majority-vote over rolling buffer (last N frames, default N=10, 70% threshold) confirms stable gestures before committing
5. **Output**: Display predicted letter or build sentence

## Limitations & Known Issues

* **Static Handshapes Only**: The current architecture extracts single-frame landmarks, limiting it to static ASL fingerspelling. It cannot process motion-based letters (like J or Z) uniquely, nor can it recognize continuous temporal sign language words and phrases.
* **Similar Gestures**: The model can sometimes confuse visually similar gestures like M and N (M is three fingers over the thumb, N is two fingers). This mirrors real-world ASL logic but requires precise hand positioning.
* **Hosting Latency**: Due to free-tier cloud hosting (Render), the service spins down after 15 minutes of inactivity. The first request after a period of dormancy will take 30-60 seconds to fulfill.
* **Future Work**: Transitioning from an MLP to an LSTM/Transformer-based sequence model. Passing a rolling window of frames into a recurrent architecture would unlock the ability to classify full, continuous signs rather than individual static letters.

## Requirements

Ensure you meet the following dependencies (see `requirements.txt` for the full list):
* `opencv-python` 4.9.0+ (for video capture and display)
* `mediapipe` 0.10.18 (for hand landmark detection)
* `tensorflow` 2.17.0+ (for neural network)
* `scikit-learn` 1.5.0+ (for label encoding)
* `pandas` 2.2.0+ (for data handling)
* `numpy` 1.23.0-1.26.x (MediaPipe relies on <2.0 versions)
* `matplotlib` 3.9.0+ (for visualization)

## Tips for Best Results

* **Lighting**: Use consistent, bright lighting
* **Background**: Plain background improves hand detection
* **Hand position**: Keep hand centered and at comfortable distance
* **Training data**: Collect 50-100 samples per gesture in varied positions
* **Gesture stability**: Hold each sign steady for ~1 second for recognition

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
