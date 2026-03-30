# Wearable_Schizophrenia_Reality_Tracking
Wearable Schizophrenia Reality-Tracking Device
Author: Sameeksha R
Status: Active Development (Aug 2025 – Present)
Hardware: STM32F446RE | MEMS Microphone | MAX86141 Biometric | BLE | Custom PCB

Overview
A wearable embedded AI device that performs real-time speech monitoring to assist in distinguishing reality from hallucination episodes in schizophrenia patients. On-device MFCC extraction and neural network inference classify speech patterns as REALITY, ILLUSION, or UNCERTAIN, and alerts caregivers via BLE.

System Architecture
[MEMS Mic] ──PDM──► [Audio Driver] ──► [Pre-emphasis]
                                              │
                                         [MFCC Engine]
                                              │
                                        [MLP Classifier]
                                              │
                        ┌─────────────────────┤
                        │                     │
                   [BLE Notify]          [SD Logger]
                   (caregiver)           (local log)
                        │
                 [MAX86141 Biometric]
                 (HR + SpO2 context)

Repository Structure
├── firmware/
│   ├── main.c              — Main processing loop (STM32F446RE)
│   └── mfcc.h              — MFCC feature extractor interface
│
└── ml_pipeline/
    └── train_classifier.py — MFCC extraction + MLP training + TFLite export

On-Device ML Pipeline
Audio Frame (512 samples, 32ms)
    │
    ▼
Pre-emphasis filter (α=0.97)
    │
    ▼
Hamming window
    │
    ▼
FFT (512-point)
    │
    ▼
Mel filterbank (26 filters, 80–7600 Hz)
    │
    ▼
Log compression
    │
    ▼
DCT → 13 MFCC coefficients
    │
    ▼
Δ + ΔΔ → 39-dim feature vector
    │
    ▼
MLP Classifier (64→32→2) → Softmax
    │
    ▼
REALITY / ILLUSION / UNCERTAIN

Episode Detection Logic

5 consecutive ILLUSION frames → episode triggered
BLE notification sent to caregiver app with confidence + heart rate + SpO2
All frames logged to SD card with timestamps


ML Training
bash# Setup
pip install numpy librosa scikit-learn tensorflow

# Add your audio data
mkdir -p data/reality data/illusion
# Place .wav files in each folder

# Train
python ml_pipeline/train_classifier.py
# Outputs: classifier.tflite + scaler_params.json

