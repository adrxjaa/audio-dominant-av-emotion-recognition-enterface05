# Audio-Dominant Multimodal Emotion Recognition (AV Fusion)

Deep learning framework for audio-visual emotion classification using feature-level fusion on the eNTERFACE’05 dataset.

---

## Overview

This project implements a multimodal emotion recognition system that combines:

- Speech-based emotion cues (MFCC features)
- Facial expression features (video frames)
- Audio-dominant embedding fusion

Final Validation Accuracy: **70.5%**  
Macro F1-score: **0.70**

---

## Repository Structure

```
audio-dominant-av-emotion-recognition/
│
├── dl_finalcode.ipynb
├── README.md
└── requirements.txt
```

The complete implementation is inside `dl_finalcode.ipynb`.

---

## Dataset

- eNTERFACE’05 Audio-Visual Emotion Dataset
- 44 participants
- 6 emotions:
  - Anger
  - Disgust
  - Fear
  - Happiness
  - Sadness
  - Surprise
- 3–5 second AVI clips

---

## Methodology

### Audio Model
- 16kHz mono extraction
- 40 MFCC + delta + delta-delta (120 features)
- 1D CNN architecture
- 64-dimensional embedding

### Video Model
- Frame sampling (3 fps)
- Haar Cascade face detection
- 48×48 grayscale frames
- CNN-based spatial modeling
- 32-dimensional embedding

### Fusion Strategy
- Feature-level fusion
- L2-normalized embeddings
- Audio-dominant formulation
- Interaction modeling
- Fully connected classifier with label smoothing

---

## Results

| Model        | Validation Accuracy | Macro F1 |
|-------------|--------------------|----------|
| Audio-only  | ~65%               | —        |
| Video-only  | Lower              | —        |
| Fusion      | **70.5%**          | **0.70** |

---

## Tech Stack

- Python
- TensorFlow
- Librosa
- OpenCV
- FFmpeg
- NumPy
- Scikit-learn

---

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Download the eNTERFACE’05 dataset.

3. Update dataset paths inside `dl_finalcode.ipynb`.

4. Run all notebook cells.

---

System Dependency:
- FFmpeg must be installed and added to PATH.
