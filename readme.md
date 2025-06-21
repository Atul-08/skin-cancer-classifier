# Skin Cancer Classifier Flask App

This is a Flask web application that uses a Convolutional Neural Network (CNN) to classify skin lesion images as **benign** or **malignant**.

## Features

- Upload skin lesion images
- Classify into benign/malignant
- Toggle between dark/light themes
- Responsive user interface

## How to Run Locally

```bash
git clone https://github.com/yourusername/skin-cancer-classifier.git
cd skin-cancer-classifier
pip install -r requirements.txt
python app.py

## 🔬 Optimization

This CNN model was optimized using a **Genetic Algorithm (GA)** for hyperparameter tuning and architecture search.  
As a result, the classification accuracy improved significantly:

- **Before Optimization:** 78.9%
- **After GA Optimization:** 91.56%

The Genetic Algorithm was applied to parameters such as:
- Learning rate
- Number of filters
- Dropout rates
- Dense layer configurations


## 🧬 Genetic Algorithm Process

1. **Initialize Population**: Random sets of hyperparameters
2. **Evaluate Fitness**: Train CNNs and calculate accuracy
3. **Select Parents**: Top-performing configurations
4. **Crossover & Mutation**: Generate new population
5. **Repeat**: Iterate until performance plateaus

This approach helped find optimal architecture + training settings to maximize classification performance.



## 🏆 Accuracy Improvement with Genetic Algorithm

| Phase              | Accuracy |
|--------------------|----------|
| Pre-Optimization   | 78.90%   |
| Post-GA Optimization | **91.56%** |

> This was achieved using genetic search for CNN hyperparameters such as dropout rate, number of filters, kernel size, and learning rate.
