# Sign Language to Text

A real-time American Sign Language (ASL) recognition system that translates hand gestures into text using computer vision and machine learning.

## Overview

This project uses **MediaPipe Hands** for landmark detection and a **Keras neural network** to classify ASL gestures. It supports the full alphabet (A-Z) plus special gestures for space and delete, enabling users to spell out words and sentences through sign language.

## Features

- **Real-time gesture recognition** via webcam
- **26 letter alphabet** (A-Z) recognition
- **Special gestures**: `space` and `del` (delete)
- **Sentence building** with prediction smoothing
- **Auto-punctuation** after pauses
- **Data collection tool** for training custom gestures

## Project Structure

```
sign_to_text/
├── collect.py          # Data collection tool - capture hand landmarks
├── training.py         # Train the MLP classifier
├── app.py              # Real-time inference with sentence building
├── eval.py             # Model evaluation (accuracy, confusion matrix)
├── test_model.h5       # Pre-trained Keras model (96.78% accuracy)
├── label_encoder.pkl   # Scikit-learn label encoder
├── data.csv            # Training dataset (57k samples from Kaggle)
└── requirements.txt    # Python dependencies
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

### Run Real-Time Recognition

**Main inference app** (sentence builder - recommended):
```bash
python app.py
```
- Builds sentences letter-by-letter
- Shows running text on screen
- Supports `space` and `del` gestures
- Auto-adds period (`.`) after 3 seconds of no hand detection
- Press **ESC** to exit

### Collect Training Data

To train on your own gestures or add new signs:

```bash
python collect.py
```

1. Position your hand in view of the webcam
2. Press **A-Z** on your keyboard to label and save the current gesture
3. Collect 50-100 samples per letter for best results
4. Data saved to `sign_data.csv`
5. Press **Q** to quit

### Train Your Own Model

After collecting data:

```bash
python training.py
```

This will:
- Load training data from `data.csv`
- Train a 3-layer MLP neural network with early stopping
- Save the model as `test_model.h5`
- Save the label encoder as `label_encoder.pkl`

### Evaluate the Model

To evaluate the current model (`test_model.h5`) against `data.csv`:

```bash
python eval.py
```

Generates:
- Overall accuracy
- Detailed classification report (precision, recall, f1-score per class)
- Confusion matrix (saved as `confusion_matrix.png`)

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

**Performance:** The default model was trained on ~57,000 diverse images from the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset) and achieves **96.78% accuracy** on the test split.

## How It Works

1. **Hand Detection:** MediaPipe Hands detects hand landmarks (21 points per hand)
2. **Feature Extraction:** Extract x, y, z coordinates → 63-dimensional vector
3. **Classification:** Neural network predicts gesture class
4. **Smoothing:** Prediction history (last 15 frames) confirms stable gestures
5. **Output:** Display predicted letter or build sentence

## Known Issues

1. **Similar gestures:** The model can sometimes confuse visually similar gestures like `M` and `N` (M is three fingers over the thumb, N is two fingers). This mirrors real-world ASL logic but requires precise hand positioning.
2. **CSV filename mismatch for collection:** `collect.py` writes to `sign_data.csv`, but `training.py` reads `data.csv`. Rename the file before training custom data.

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

- [ ] Fix feature ordering bug
- [ ] Add word-level recognition
- [ ] Support dynamic gestures (motion-based signs)
- [ ] Add grammar/autocorrect
- [ ] Mobile app deployment
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
