# Power Transformer Fault Diagnosis System

## Overview

This project presents a data-driven approach to fault diagnosis in power transformers using machine learning techniques applied to real-world engineering data.

Reliable fault detection is critical for preventing power outages, reducing economic loss, and maintaining grid stability in modern power systems.

This work is based on the research paper:

📄 https://www.sciencedirect.com/science/article/pii/S2590123026010091

---

## Problem

Power transformers are critical components in electrical infrastructure, and failures can lead to severe operational and financial consequences.

However, real-world fault detection is challenging due to:

* Noisy measurement signals
* Complex system behavior
* Variability in operating conditions

Traditional methods often struggle to generalize across different environments and datasets.

---

## Dataset

* Transformer diagnostic data (e.g., sensor readings, operational signals)
* Multi-class fault classification

The dataset represents real-world variability in transformer conditions, making the problem more challenging than controlled benchmark datasets.

---

## Fault Types

* Thermal faults
* Electrical faults
* Insulation degradation
* Partial discharge conditions

---

## System Approach

### Models

* Machine learning models for fault classification
* Feature-based learning approach

### Pipeline

* Data preprocessing and cleaning
* Feature engineering
* Model training and validation
* Performance evaluation

---

## Challenges

* Noise and inconsistencies in measurement data
* Class imbalance across fault categories
* Generalization across different transformer conditions

---

## Results & Observations

* Machine learning models can effectively detect fault patterns in structured data
* Performance is highly dependent on feature quality and preprocessing

> Most errors observed were not due to model limitations, but due to variability and noise in real-world transformer data.

This highlights that fault diagnosis is not purely a modeling problem, but a system-level challenge involving data quality, preprocessing, and robustness.

This suggests that improving performance requires better integration of sensing, preprocessing, and modeling rather than focusing solely on model architecture.

---

## Visual Results

### Feature Importance by Dissolved Gas

<img width="3547" height="3720" alt="image" src="https://github.com/user-attachments/assets/d4ed8fb9-8fb6-4feb-a411-3ca8b4301388" />

---

## Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib

---

## Repository Structure

```
├── notebooks/
├── src/
├── data/
├── results/
├── requirements.txt
└── README.md
```

---

## Research Context

Machine learning has become a powerful tool for fault diagnosis in electrical systems, enabling predictive maintenance and improving system reliability.

Real-world applications require models that can handle noisy and variable data, rather than relying solely on ideal conditions.

---

## Citation

If you use this work, please cite:

```
Raif Tanjim,
"Data-Driven Fault Diagnosis Framework for Power Transformers",
2026.
```

📄 Paper:
https://www.sciencedirect.com/science/article/pii/S2590123026010091

---

## Future Work

* Improve robustness under varying operating conditions
* Integrate real-time monitoring systems
* Explore deep learning-based fault detection

---

## Author

Raif Tanjim
AI & Robotics | Perception & Real-World Adaptive Systems
