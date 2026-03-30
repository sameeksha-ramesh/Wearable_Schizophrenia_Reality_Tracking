/**
 * @file    mfcc.h
 * @brief   MFCC Feature Extractor — Wearable Schizophrenia Tracker
 * @author  Sameeksha R
 *
 * Mel-Frequency Cepstral Coefficients extraction pipeline:
 *   PCM frame → Hamming window → FFT → Mel filterbank → Log → DCT → MFCCs
 */
 
#ifndef MFCC_H
#define MFCC_H
 
#include <stdint.h>
 
#define MFCC_MAX_COEFFS   40
#define MFCC_NUM_FILTERS  26
#define MEL_LOW_FREQ      80.0f
#define MEL_HIGH_FREQ     7600.0f
#define LOG_FLOOR         1e-10f
 
/**
 * @brief  Initialise MFCC processor.
 * @param  sample_rate   Audio sample rate in Hz.
 * @param  frame_size    Number of samples per frame (power of 2).
 * @param  num_coeffs    Number of MFCC coefficients to compute.
 */
void mfcc_init(uint32_t sample_rate, uint32_t frame_size, uint32_t num_coeffs);
 
/**
 * @brief  Compute MFCC features from one audio frame.
 * @param  frame      Input PCM frame (int16_t, length = frame_size).
 * @param  frame_len  Number of samples.
 * @param  mfcc_out   Output buffer for MFCC coefficients (float[num_coeffs]).
 * @param  num_coeffs Number of coefficients to compute.
 */
void mfcc_compute(const int16_t *frame, uint32_t frame_len,
                  float *mfcc_out, uint32_t num_coeffs);
 
/**
 * @brief  Convert frequency in Hz to Mel scale.
 * @param  hz  Frequency in Hz.
 * @return Mel scale value.
 */
float hz_to_mel(float hz);
 
/**
 * @brief  Convert Mel scale value to Hz.
 * @param  mel  Mel scale value.
 * @return Frequency in Hz.
 */
float mel_to_hz(float mel);
 
#endif /* MFCC_H */
