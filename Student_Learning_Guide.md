# Speech Feature Extraction - Student Learning Guide

## Course: Smart and Connected Health
## Project: PD vs TBI Classification using Speech Analysis

---

# PART 1: INTRODUCTION

## What is this project about?
- We have audio recordings from Parkinson's Disease (PD) and Traumatic Brain Injury (TBI) patients
- Goal: Extract speech features → Train ML model → Classify PD vs TBI
- Your job: Learn how to extract features from audio files

## Why speech features?
- Both PD and TBI affect speech patterns
- Voice changes can be measured objectively
- Non-invasive way to help diagnosis

---

# PART 2: PREREQUISITES

## What you need to know:
- Basic Python programming
- How to use Google Colab
- Basic understanding of arrays and loops

## What you will learn:
- How audio files work
- How to extract meaningful features from speech
- Using librosa and parselmouth libraries

---

# PART 3: LEARNING PATH

## Week 1: Basics

### Day 1-2: Understand Audio Basics
**Topics:**
- What is a .wav file?
- What is sample rate?
- What is a waveform?

**Resources:**
- Read: https://pudding.cool/2018/02/waveforms/
- Watch: "How Digital Audio Works" on YouTube

### Day 3-4: Setup Environment
**Tasks:**
1. Open Google Colab
2. Install libraries:
```python
!pip install librosa praat-parselmouth
!apt-get install -y ffmpeg
```
3. Load a sample audio file
4. Play audio in Colab

### Day 5-7: Run Starter Code
**Tasks:**
- Use the provided starter code
- Extract features from one audio file
- Understand what each feature means

---

## Week 2: Deep Dive into Libraries

### Day 1-3: Learn LIBROSA

**Documentation:** https://librosa.org/doc/latest/index.html

**Topics to cover:**
| Topic | Function |
|-------|----------|
| Loading audio | `librosa.load()` |
| MFCCs | `librosa.feature.mfcc()` |
| Spectral features | `spectral_centroid`, `bandwidth`, `rolloff` |
| Zero crossing rate | `zero_crossing_rate()` |
| RMS energy | `rms()` |
| Silence detection | `librosa.effects.split()` |

**Tutorial:** https://librosa.org/doc/latest/tutorial.html

### Day 4-6: Learn PARSELMOUTH

**Documentation:** https://parselmouth.readthedocs.io/

**Topics to cover:**
| Topic | Description |
|-------|-------------|
| Loading audio | `parselmouth.Sound()` |
| Pitch extraction | Fundamental frequency |
| Jitter | Pitch stability |
| Shimmer | Amplitude stability |
| HNR | Harmonics-to-noise ratio |
| Formants | F1, F2, F3 |

**Praat Manual:** https://www.fon.hum.uva.nl/praat/manual/

### Day 7: Understand the Science

**Read about what each feature means clinically:**

- **Jitter & Shimmer:** Voice disorder indicators
  - https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html

- **MFCCs:** Voice fingerprint
  - https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd

- **HNR:** Voice clarity measurement
  - Higher = clearer voice
  - Lower = breathy/hoarse

---

# PART 4: FEATURE CATEGORIES

## Category 1: VOICE QUALITY FEATURES (parselmouth)

| Feature | What It Measures | Clinical Use |
|---------|------------------|--------------|
| Pitch (F0) | Voice frequency | Gender, age |
| Pitch Std | Pitch variation | Monotone voice |
| Jitter (local) | Pitch instability | PD indicator |
| Jitter (rap) | Relative pitch perturb | Voice tremor |
| Jitter (ppq5) | 5-point pitch perturb | Voice tremor |
| Shimmer (local) | Volume instability | Weak voice |
| Shimmer (apq3) | 3-point amplitude | Voice tremor |
| Shimmer (apq5) | 5-point amplitude | Voice tremor |
| HNR | Voice clarity | Breathiness |
| Formants (F1-F4) | Vocal tract shape | Articulation |

## Category 2: SPECTRAL FEATURES (librosa)

| Feature | What It Measures | Function |
|---------|------------------|----------|
| MFCCs (1-13) | Voice texture | `mfcc()` |
| Delta MFCCs | MFCC change over time | `delta()` |
| Spectral Centroid | Brightness of sound | `spectral_centroid()` |
| Spectral Bandwidth | Frequency spread | `spectral_bandwidth()` |
| Spectral Rolloff | Frequency cutoff | `spectral_rolloff()` |
| Spectral Contrast | Peak vs valley diff | `spectral_contrast()` |
| Spectral Flatness | Noise vs tone | `spectral_flatness()` |
| Chroma Features | Pitch class energy | `chroma_stft()` |
| Mel Spectrogram | Frequency over time | `melspectrogram()` |

## Category 3: TEMPORAL FEATURES (librosa)

| Feature | What It Measures | Function |
|---------|------------------|----------|
| Duration | Total length | `get_duration()` |
| Zero Crossing Rate | Sign changes | `zero_crossing_rate()` |
| RMS Energy | Loudness | `rms()` |
| Pause Count | Number of silences | `effects.split()` |
| Pause Duration | Total silence time | `effects.split()` |
| Speech Rate | Speaking speed | Manual calc |
| Tempo | Beat estimation | `beat.tempo()` |

---

# PART 5: CODE EXAMPLES

## Example 1: Load Audio
```python
import librosa
import parselmouth

# Method 1: librosa (for spectral features)
audio, sr = librosa.load("file.wav", sr=16000)

# Method 2: parselmouth (for voice quality)
sound = parselmouth.Sound("file.wav")
```

## Example 2: Extract Pitch
```python
from parselmouth.praat import call

sound = parselmouth.Sound("file.wav")
pitch = call(sound, "To Pitch", 0.0, 75, 500)
pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
pitch_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")

print(f"Pitch: {pitch_mean} Hz")
```

## Example 3: Extract Jitter & Shimmer
```python
from parselmouth.praat import call

sound = parselmouth.Sound("file.wav")
point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

print(f"Jitter: {jitter}")
print(f"Shimmer: {shimmer}")
```

## Example 4: Extract HNR
```python
from parselmouth.praat import call

sound = parselmouth.Sound("file.wav")
harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
hnr = call(harmonicity, "Get mean", 0, 0)

print(f"HNR: {hnr} dB")
```

## Example 5: Extract MFCCs
```python
import librosa
import numpy as np

audio, sr = librosa.load("file.wav", sr=16000)
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mfcc_means = np.mean(mfccs, axis=1)

for i in range(13):
    print(f"MFCC {i+1}: {mfcc_means[i]}")
```

## Example 6: Extract Spectral Features
```python
import librosa
import numpy as np

audio, sr = librosa.load("file.wav", sr=16000)

centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

print(f"Centroid: {centroid} Hz")
print(f"Bandwidth: {bandwidth} Hz")
print(f"Rolloff: {rolloff} Hz")
print(f"Flatness: {flatness}")
```

## Example 7: Detect Pauses
```python
import librosa

audio, sr = librosa.load("file.wav", sr=16000)
non_silent = librosa.effects.split(audio, top_db=25)

num_pauses = len(non_silent) - 1
duration = librosa.get_duration(y=audio, sr=sr)
speech_time = sum([(end - start) / sr for start, end in non_silent])
pause_time = duration - speech_time

print(f"Number of pauses: {num_pauses}")
print(f"Pause time: {pause_time} seconds")
```

## Example 8: Extract Formants (Advanced)
```python
from parselmouth.praat import call

sound = parselmouth.Sound("file.wav")
formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)

f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")

print(f"F1: {f1} Hz")
print(f"F2: {f2} Hz")
print(f"F3: {f3} Hz")
```

---

# PART 6: DOCUMENTATION LINKS

## Primary Documentation

| Library | Link |
|---------|------|
| librosa | https://librosa.org/doc/latest/index.html |
| librosa features | https://librosa.org/doc/latest/feature.html |
| parselmouth | https://parselmouth.readthedocs.io/ |
| Praat manual | https://www.fon.hum.uva.nl/praat/manual/ |

## Tutorials

| Topic | Link |
|-------|------|
| librosa tutorial | https://librosa.org/doc/latest/tutorial.html |
| MFCCs explained | https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd |
| Audio basics | https://pudding.cool/2018/02/waveforms/ |
| Speech processing | https://speechprocessingbook.aalto.fi/ |

## Research Papers

| Topic | Link |
|-------|------|
| PD voice detection | https://www.frontiersin.org/articles/10.3389/frai.2023.1084001/full |
| TBI speech markers | https://neuro.jmir.org/2025/1/e64624 |
| Jitter & Shimmer | https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html |

---

# PART 7: ASSIGNMENTS

## Assignment 1: Run the Starter Code
- Load the provided audio file
- Run the feature extraction code
- Print all 23 features
- **Deliverable:** Screenshot of output

## Assignment 2: Add 5 New Features
Choose from this list and add to the code:
- [ ] Spectral bandwidth
- [ ] Spectral rolloff
- [ ] Spectral flatness
- [ ] RMS energy
- [ ] Formants (F1, F2, F3)
- [ ] Additional jitter variants (rap, ppq5)
- [ ] Additional shimmer variants (apq3, apq5)
- [ ] Delta MFCCs
- [ ] Tempo

**Deliverable:** Updated code with 5 new features

## Assignment 3: Process Multiple Files
- Create a loop to process all audio files in a folder
- Save results to a CSV file
- **Deliverable:** CSV file with features for all patients

## Assignment 4: Feature Analysis
- Calculate mean and std of each feature for PD vs TBI
- Which features show the biggest difference?
- **Deliverable:** Short report with findings

---

# PART 8: QUICK REFERENCE CHEAT SHEET

```
┌──────────────────────────────────────────────────────────────┐
│                FEATURE EXTRACTION CHEAT SHEET                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  STEP 1: INSTALL                                             │
│  !pip install librosa praat-parselmouth                      │
│  !apt-get install -y ffmpeg                                  │
│                                                              │
│  STEP 2: CONVERT M4A TO WAV                                  │
│  !ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav            │
│                                                              │
│  STEP 3: LOAD AUDIO                                          │
│  audio, sr = librosa.load("file.wav", sr=16000)  # librosa  │
│  sound = parselmouth.Sound("file.wav")       # parselmouth  │
│                                                              │
│  STEP 4: EXTRACT FEATURES                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ PARSELMOUTH (voice quality)                            │ │
│  │ • Pitch:   call(sound, "To Pitch", ...)               │ │
│  │ • Jitter:  call(pp, "Get jitter (local)", ...)        │ │
│  │ • Shimmer: call([sound, pp], "Get shimmer...", ...)   │ │
│  │ • HNR:     call(harmonicity, "Get mean", ...)         │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ LIBROSA (spectral & temporal)                          │ │
│  │ • MFCCs:    librosa.feature.mfcc(y, sr, n_mfcc=13)    │ │
│  │ • Centroid: librosa.feature.spectral_centroid(y, sr)  │ │
│  │ • ZCR:      librosa.feature.zero_crossing_rate(y)     │ │
│  │ • Pauses:   librosa.effects.split(y, top_db=25)       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  STEP 5: SAVE TO CSV                                         │
│  df = pd.DataFrame(all_features)                             │
│  df.to_csv("features.csv", index=False)                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

# QUESTIONS?

Contact your mentor if you have any questions!
