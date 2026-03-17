# Disease Mortality Prediction: ML Tutorial

A comprehensive tutorial demonstrating how to build and train machine learning models using **scikit-learn** and **PyTorch** to predict disease mortality rates.

## Overview

This project provides a complete machine learning pipeline including:
- Synthetic medical dataset generation
- Exploratory Data Analysis (EDA)
- Model development with scikit-learn (Random Forest)
- Deep learning implementation with PyTorch (Neural Networks)
- Performance comparison and evaluation
- Real-world prediction examples

## Features

✅ Data preprocessing and normalization
✅ Train-test split with stratification
✅ Random Forest classifier with feature importance
✅ PyTorch neural network with dropout regularization
✅ Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
✅ Visualization of results and comparisons
✅ GPU support for PyTorch

## Prerequisites

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
```

## Installation

### Option 1: Using pip
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

### Option 2: Using Anaconda
```bash
conda install numpy pandas matplotlib seaborn scikit-learn pytorch torchvision torchaudio
```

## Usage

1. Open the Jupyter notebook
```bash
jupyter notebook disease_prediction_tutorial.ipynb
```

2. Run all cells sequentially to:
   - Generate synthetic dataset
   - Perform EDA
   - Train both models
   - Compare performance
   - Make predictions on new data

## Project Structure

```
disease-prediction-ml/
├── disease_prediction_tutorial.ipynb  # Main tutorial notebook
├── README.md                          # This file
└── .gitignore                         # Git ignore file
```

## Notebook Sections

1. **Import Required Libraries** - Set up dependencies
2. **Generate Synthetic Disease Dataset** - Create 1000 patient records
3. **Data Preprocessing and Exploration** - EDA and feature statistics
4. **Split Data into Training and Testing Sets** - 80-20 split
5. **Build and Train Scikit-learn Model** - Random Forest classifier
6. **Build and Train PyTorch Neural Network** - Deep learning model
7. **Model Evaluation and Comparison** - Performance metrics and visualizations
8. **Make Predictions on New Data** - Real-world usage examples

## Model Architecture

### Random Forest
- 100 trees
- Max depth: 10
- Optimized for interpretability

### PyTorch Neural Network
- Input: 9 features
- Hidden layers: 64 → 32 → 16 → 8 neurons
- Activation: ReLU
- Dropout: 0.2-0.3 for regularization
- Output: Binary classification (Low/High Risk)
- Optimizer: Adam (lr=0.001)
- Loss: Cross Entropy

## Results

Both models achieve strong performance on the test set with:
- High accuracy rates
- Good precision and recall balance
- Strong ROC-AUC scores

Random Forest excels at feature importance analysis, while PyTorch offers flexibility for more complex architectures.

## Key Takeaways

1. **Scikit-learn** is ideal for quick ML implementation on structured data
2. **PyTorch** provides flexibility for custom deep learning solutions
3. Always compare multiple models before production deployment
4. Proper data preprocessing is crucial for model performance
5. Regular evaluation with multiple metrics prevents overfitting

## Best Practices

- Normalize/scale features before training
- Use stratified splits for balanced data
- Evaluate with multiple metrics (not just accuracy)
- Compare models systematically
- Document all preprocessing steps
- Use cross-validation for robust results

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- Cross-validation implementation
- Class imbalance handling (if applicable)
- Advanced feature engineering
- Ensemble methods
- Model interpretability (SHAP, LIME)
- Productionization and deployment

## Author

Created as a comprehensive ML learning tutorial

## License

Open source - feel free to use and modify

## Contributing

Contributions are welcome! Feel free to fork, modify, and submit pull requests.
