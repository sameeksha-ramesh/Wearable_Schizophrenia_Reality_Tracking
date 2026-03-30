"""
train_classifier.py — Schizophrenia Speech Pattern Classifier
Author : Sameeksha R
 
Trains a lightweight speech classifier to distinguish:
  - REALITY : normal speech patterns
  - ILLUSION : disorganized/hallucination-linked speech patterns
 
Pipeline:
  Audio files → MFCC extraction → Feature normalisation
  → MLP classifier → TFLite export (for STM32 deployment)
 
Dataset structure expected:
  data/
    reality/   *.wav files of normal speech
    illusion/  *.wav files of disorganized speech
 
Usage:
    pip install numpy librosa scikit-learn tensorflow
    python train_classifier.py
"""
 
import os
import numpy as np
import librosa
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
 
# ── Configuration ──────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
FRAME_DURATION = 0.032        # 32 ms frames
HOP_DURATION   = 0.016        # 50% overlap
N_MFCC         = 13
DATA_DIR       = "data"
MODEL_PATH     = "classifier.tflite"
SCALER_PATH    = "scaler_params.json"
 
CLASSES = ["reality", "illusion", "uncertain"]
LABEL_MAP = {cls: i for i, cls in enumerate(CLASSES)}
 
# ── Feature extraction ─────────────────────────────────────────────────
def extract_mfcc(filepath: str) -> np.ndarray:
    """
    Load audio, apply pre-emphasis, extract per-frame MFCCs.
    Returns mean + std of MFCC coefficients (shape: 2 * N_MFCC).
    """
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
 
    # Pre-emphasis filter
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
 
    frame_len = int(FRAME_DURATION * sr)
    hop_len   = int(HOP_DURATION   * sr)
 
    # MFCC extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=frame_len, hop_length=hop_len)
 
    # Delta and delta-delta (dynamic features)
    delta  = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
 
    # Aggregate: mean + std over time → fixed-length feature vector
    feats = np.concatenate([
        mfccs.mean(axis=1),  mfccs.std(axis=1),
        delta.mean(axis=1),  delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ])
    return feats
 
# ── Dataset loading ────────────────────────────────────────────────────
def load_dataset() -> tuple:
    X, y = [], []
    for cls in ["reality", "illusion"]:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"[WARN] Missing: {cls_dir} — skipping")
            continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".wav"):
                continue
            fpath = os.path.join(cls_dir, fname)
            try:
                feats = extract_mfcc(fpath)
                X.append(feats)
                y.append(LABEL_MAP[cls])
            except Exception as e:
                print(f"[WARN] {fpath}: {e}")
 
    print(f"[DATA] Loaded {len(X)} samples  "
          f"(reality: {y.count(0)}, illusion: {y.count(1)})")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
 
# ── Train model ────────────────────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray):
    import tensorflow as tf
    from tensorflow import keras
 
    # Normalise
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # Save scaler params for firmware
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(SCALER_PATH, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    print(f"[SCALER] Saved to {SCALER_PATH}")
 
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
 
    # One-hot encode (2 classes: reality + illusion)
    num_classes = 2
    y_train_oh  = keras.utils.to_categorical(y_train, num_classes)
    y_test_oh   = keras.utils.to_categorical(y_test,  num_classes)
 
    # Model: lightweight MLP suitable for STM32 deployment
    input_dim = X_train.shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax'),
    ], name="SchizoClassifier")
 
    model.summary()
 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
 
    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train_oh,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
 
    # Evaluate
    y_pred_probs = model.predict(X_test)
    y_pred       = y_pred_probs.argmax(axis=1)
    print("\n[EVAL] Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["reality", "illusion"]))
    print("[EVAL] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
 
    # Export to TFLite (int8 quantised for STM32 deployment)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
 
    with open(MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"\n[MODEL] TFLite model saved: {MODEL_PATH} "
          f"({len(tflite_model)/1024:.1f} KB)")
 
    return model
 
# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Schizophrenia Speech Classifier — Sameeksha R ===\n")
    X, y = load_dataset()
 
    if len(X) == 0:
        print("[ERROR] No data found. Add .wav files to data/reality/ and data/illusion/")
        exit(1)
 
    model = train(X, y)
    print("\n[DONE] Model trained and exported for STM32 deployment.")
