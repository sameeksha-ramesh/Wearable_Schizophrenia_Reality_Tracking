/**
 * @file    main.c
 * @brief   Wearable Schizophrenia Reality-Tracking Device — Firmware
 * @author  Sameeksha R
 *
 * Real-time speech monitoring wearable for schizophrenia patients.
 * Captures audio via MEMS microphone, extracts MFCC features,
 * runs an on-device classifier to flag potential hallucination episodes,
 * logs events and alerts caregiver via BLE.
 *
 * Pipeline:
 *  Audio (PDM mic) → ADC buffer → Pre-emphasis → Frame → FFT
 *  → Mel filterbank → MFCC → Classifier → Reality/Illusion label
 *  → BLE notify + SD log
 *
 * Hardware : STM32L476RG (low-power Cortex-M4 + FPU)
 * Peripherals: IMP34DT05 MEMS mic (PDM), MAX86141 biometric sensor,
 *              BLE module (HM-10), SD card (SPI), OLED (I2C)
 */
 
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "audio_driver.h"
#include "mfcc.h"
#include "classifier.h"
#include "ble_driver.h"
#include "sd_logger.h"
#include "biometric_driver.h"
 
/* ── Configuration ──────────────────────────────────────────────────── */
#define SAMPLE_RATE_HZ       16000
#define FRAME_SIZE           512          /* ~32ms at 16kHz */
#define HOP_SIZE             256          /* 50% overlap */
#define NUM_MFCC_COEFFS      13
#define CLASSIFIER_THRESHOLD 0.72f        /* Confidence to flag episode */
#define LOG_INTERVAL_MS      1000
 
/* ── Labels ─────────────────────────────────────────────────────────── */
#define LABEL_REALITY        0
#define LABEL_ILLUSION       1
#define LABEL_UNCERTAIN      2
 
static const char* label_str[] = { "REALITY", "ILLUSION", "UNCERTAIN" };
 
/* ── Buffers ────────────────────────────────────────────────────────── */
static int16_t  audio_buf[FRAME_SIZE];
static float    mfcc_features[NUM_MFCC_COEFFS];
static float    classifier_out[3];        /* Softmax probabilities */
 
/* ── Episode tracking ───────────────────────────────────────────────── */
static uint32_t illusion_streak   = 0;
static uint32_t episode_count     = 0;
#define EPISODE_TRIGGER_FRAMES     5      /* 5 consecutive illusion frames */
 
/* ════════════════════════════════════════════════════════════════════ */
 
/**
 * @brief  Apply pre-emphasis filter to reduce low-freq noise.
 *         y[n] = x[n] - 0.97 * x[n-1]
 */
static void pre_emphasis(int16_t *buf, int len)
{
    for (int i = len - 1; i > 0; i--)
        buf[i] = (int16_t)(buf[i] - 0.97f * buf[i - 1]);
}
 
/**
 * @brief  Compute short-time energy of frame (speech activity detector).
 * @return Energy value (higher = more speech activity)
 */
static float frame_energy(const int16_t *buf, int len)
{
    float energy = 0.0f;
    for (int i = 0; i < len; i++)
        energy += (float)buf[i] * buf[i];
    return energy / len;
}
 
/**
 * @brief  Classify result into episode label.
 */
static int classify_label(const float *probs)
{
    if (probs[LABEL_ILLUSION] >= CLASSIFIER_THRESHOLD)
        return LABEL_ILLUSION;
    if (probs[LABEL_REALITY]  >= CLASSIFIER_THRESHOLD)
        return LABEL_REALITY;
    return LABEL_UNCERTAIN;
}
 
/**
 * @brief  Send BLE notification to caregiver app.
 */
static void notify_caregiver(uint32_t episode_id,
                              float confidence,
                              biometric_data_t *bio)
{
    char payload[128];
    snprintf(payload, sizeof(payload),
             "{\"ep\":%lu,\"conf\":%.2f,\"hr\":%u,\"spo2\":%u}",
             episode_id, confidence,
             bio->heart_rate, bio->spo2);
 
    ble_notify(payload);
    printf("[BLE] Notified caregiver: %s\n", payload);
}
 
/* ════════════════════════════════════════════════════════════════════ */
void app_main(void)
{
    printf("\n=== Wearable Schizophrenia Reality Tracker — Sameeksha R ===\n");
    printf("Sample rate : %d Hz\n", SAMPLE_RATE_HZ);
    printf("Frame size  : %d samples (%.1f ms)\n",
           FRAME_SIZE, (float)FRAME_SIZE / SAMPLE_RATE_HZ * 1000.0f);
    printf("MFCC coeffs : %d\n\n", NUM_MFCC_COEFFS);
 
    /* Init all peripherals */
    audio_driver_init(SAMPLE_RATE_HZ);
    mfcc_init(SAMPLE_RATE_HZ, FRAME_SIZE, NUM_MFCC_COEFFS);
    classifier_init();
    ble_init();
    sd_logger_init();
    biometric_driver_init();
 
    biometric_data_t bio;
    uint32_t frame_count = 0;
 
    /* ── Main processing loop ── */
    while (1) {
        /* 1. Capture audio frame */
        audio_driver_read(audio_buf, FRAME_SIZE);
        frame_count++;
 
        /* 2. Voice activity detection — skip silent frames */
        float energy = frame_energy(audio_buf, FRAME_SIZE);
        if (energy < 1e6f) {
            illusion_streak = 0;
            continue;
        }
 
        /* 3. Pre-emphasis */
        pre_emphasis(audio_buf, FRAME_SIZE);
 
        /* 4. Extract MFCC features */
        mfcc_compute(audio_buf, FRAME_SIZE, mfcc_features, NUM_MFCC_COEFFS);
 
        /* 5. Run classifier */
        classifier_infer(mfcc_features, NUM_MFCC_COEFFS, classifier_out);
        int label = classify_label(classifier_out);
 
        /* 6. Read biometrics */
        biometric_driver_read(&bio);
 
        /* 7. Episode detection logic */
        if (label == LABEL_ILLUSION) {
            illusion_streak++;
            if (illusion_streak == EPISODE_TRIGGER_FRAMES) {
                episode_count++;
                printf("\n[EPISODE] #%lu detected! conf=%.2f  HR=%u  SpO2=%u%%\n",
                       episode_count,
                       classifier_out[LABEL_ILLUSION],
                       bio.heart_rate, bio.spo2);
 
                notify_caregiver(episode_count,
                                 classifier_out[LABEL_ILLUSION],
                                 &bio);
            }
        } else {
            illusion_streak = 0;
        }
 
        /* 8. SD log every frame with label */
        sd_logger_write(frame_count,
                        label_str[label],
                        classifier_out[LABEL_ILLUSION],
                        bio.heart_rate,
                        bio.spo2);
 
        /* 9. Console output */
        printf("[FRAME %lu] label=%-9s  illusion=%.3f  reality=%.3f  "
               "HR=%u  SpO2=%u%%\n",
               frame_count,
               label_str[label],
               classifier_out[LABEL_ILLUSION],
               classifier_out[LABEL_REALITY],
               bio.heart_rate,
               bio.spo2);
    }
}
