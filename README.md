# Hand Gesture Recognition Using Wi-Fi RSSI Data

This repository contains the implementation of a hand gesture recognition system using Wi-Fi Received Signal Strength Indicator (RSSI) data. The project leverages machine learning techniques to classify gestures such as `Swipe`, `Push-Pull`, and `Circular`.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Dataset](#dataset)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

Wi-Fi-based gesture recognition explores the potential of RSSI data for non-invasive and touchless human activity recognition. This system:
- Captures Wi-Fi RSSI data during hand gestures.
- Preprocesses the data to ensure temporal continuity and remove noise.
- Applies machine learning models to classify gestures with high accuracy.

## Features

- RSSI data collection using Wi-Fi scanning APIs.
- Data preprocessing, including resampling, smoothing, and interpolation.
- Exploratory Data Analysis (EDA) with visualizations.
- Machine learning pipelines using models like Logistic Regression, Random Forest, SVM, XGBoost, and Stacking Ensemble.
- Comprehensive evaluation with metrics like accuracy, precision, recall, and F1-score.
- Modular codebase for easy extension and experimentation.
- Visualization tools for dataset inspection and model performance.
- Jupyter notebooks for step-by-step experimentation.

## Requirements

- Python 3.8+
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `tqdm`, `seaborn`, `jupyter`
- LaTeX for report generation (optional)
- Wi-Fi enabled device for RSSI data collection (if collecting new data)

## Dataset

The dataset consists of Wi-Fi RSSI readings captured while performing specific hand gestures in proximity to a Wi-Fi-enabled device. Each gesture class (e.g., swipe, push-pull, circular) is represented by multiple samples collected under varied conditions.  
**Note:** If you wish to use your own data, use the data collection scripts provided in the repository. Example data files and collection instructions are available in the `data/` directory.

## Usage

1. **Clone the repository**
    ```sh
    git clone https://github.com/RoYKalyan/Hand-Gesture-Recognition.git
    cd Hand-Gesture-Recognition
    ```

2. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare data**
    - Place your RSSI data files in the `data/` directory or use the provided samples.

4. **Run exploratory analysis**
    - Open and run `notebooks/EDA.ipynb` for data visualization and understanding.

5. **Train models**
    - Use `notebooks/model_training.ipynb` to train and evaluate classifiers.

6. **Evaluate results**
    - Results (confusion matrices, metrics, etc.) will be generated in the `results/` directory.

## Project Structure

```
Hand-Gesture-Recognition/
├── data/                # Collected and sample RSSI data
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── src/                 # Source code for data processing and modeling
├── results/             # Output results, figures, and metrics
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License information
```

## Results

- Preliminary experiments show gesture classification accuracies exceeding 90% for well-defined gesture classes.
- Detailed evaluation metrics and plots are available in the `results/` directory.
- For experiment-specific results, see the relevant Jupyter notebooks.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for suggestions, bug fixes, or enhancements.  
To contribute:
1. Fork the repository
2. Create a new branch
3. Make your changes with clear commit messages
4. Open a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
For questions or collaboration, contact [RoYKalyan](https://github.com/RoYKalyan).
