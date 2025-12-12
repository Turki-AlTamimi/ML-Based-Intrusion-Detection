# Intrusion Detection System (IDS) - Machine Learning Pipeline

A comprehensive network intrusion detection system that leverages multiple machine learning classifiers to identify cyber attacks from network port statistics. This project provides automated model training, evaluation, and extensive performance visualization for 6 network traffic classes.

---

## 🎯 Overview

This IDS pipeline processes network flow data to classify traffic into normal and attack categories. It compares 7 individual classifiers plus a weighted ensemble method, generates detailed performance metrics, and creates publication-ready visualizations. The system supports configurable test/train splits and exports results in multiple formats.

---

## ✨ Features

- **Multi-Classifier Framework**: Compares 8 ML algorithms including Gradient Boosting, Neural Networks, Random Forest, SVM, and Ensemble Voting
- **Dual Test Split Analysis**: Automatically evaluates models on both 20% and 50% test data splits
- **Comprehensive Metrics**: Calculates TPR, FPR, Accuracy, F1-Score, and Precision per-class and averaged
- **Rich Visualizations**: Generates bar charts, heatmaps, radar charts, and summary reports
- **Cross-Platform**: Supports both Unix/Linux and Windows execution
- **Publication-Ready**: High-resolution (300 DPI) plots saved automatically
- **Automated Pipeline**: Single command runs the entire workflow from data to visualizations

---

## 📁 Project Structure

```
IDS-Project/
├── IDS.py                      # Main pipeline orchestrator
├── Classify.py                 # Core classifier implementation
├── analyze_results.py          # Compare 20% vs 50% test splits
├── export_results_to_csv.py    # Export metrics to CSV
├── Visualize_Results.py        # Generate advanced visualizations
├── run_ids_unix.sh            # Unix/Linux execution script
├── run_ids_win.bat            # Windows execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Turki-AlTamimi/ML-Based-Intrusion-Detection.git
cd ML-Based-Intrusion-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset (`data.csv`) in the project root

### Running the Pipeline

**On Linux/Unix/Mac:**
```bash
chmod +x run_ids_unix.sh
./run_ids_unix.sh
```

**On Windows:**
```bash
run_ids_win.bat
```

**Manual Execution (if needed):**
```bash
python IDS.py -csv data.csv
python analyze_results.py
python export_results_to_csv.py
python Visualize_Results.py -results data.csv_50.results.txt
```

---

## 📊 Classifiers Implemented

| Classifier | Description | Key Parameters |
|------------|-------------|----------------|
| **HistGradientBoosting** | State-of-the-art gradient boosting | `class_weight='balanced'` |
| **Neural Networks** | Multi-layer perceptron | 2 hidden layers, L-BFGS solver |
| **Random Forest** | 100 trees with isotonic calibration | `class_weight='balanced'` |
| **AdaBoost** | Adaptive boosting with calibration | Isotonic calibration |
| **Gradient Boosting** | Classical gradient boosting | Default sklearn parameters |
| **SVM** | RBF kernel support vector machine | `probability=True` |
| **Naive Bayes** | Gaussian Naive Bayes with calibration | Isotonic calibration |
| **Weighted Ensemble** | Soft voting ensemble | Custom weights [0.25, 0.01, 0.02, 0.25, 0.18, 0.09, 0.05] |

---

## 📈 Output Files

### Results
- `data.csv_20.results.txt` - Metrics for 20% test split
- `data.csv_50.results.txt` - Metrics for 50% test split
- `combined_results.csv` - Consolidated metrics in CSV format

### Visualizations (saved in `plots/` directory)

| File | Description |
|------|-------------|
| `compare_tpr_20_50.png` | TPR comparison bar chart |
| `compare_fpr_20_50.png` | FPR comparison bar chart |
| `compare_accuracy_20_50.png` | Accuracy comparison |
| `compare_f1_20_50.png` | F1-Score comparison |
| `classifier_comparison.png` | Average metrics across all classifiers |
| `performance_heatmaps.png` | Per-class performance heatmaps |
| `radar_charts_top4.png` | Radar charts for top 4 classifiers |
| `class_difficulty_analysis.png` | Attack detection difficulty ranking |
| `summary_report.png` | Comprehensive summary statistics |

---

## 🗂️ Dataset Format

### Expected Input
- **File**: `data.csv` (or specify via `-csv` flag)
- **Format**: Comma-separated values
- **Rows**: Network port statistics records
- **Columns**: 32 features + 1 label column

### Classes (6 categories)
| ID | Class | Description |
|----|-------|-------------|
| 0 | Normal | Normal network traffic |
| 1 | Blackhole | Blackhole attack |
| 2 | TCP-SYN | TCP-SYN flood attack |
| 3 | PortScan | Port scanning activity |
| 4 | Diversion | Traffic diversion attack |
| 5 | Overflow | Buffer overflow attack |

### Feature List
- Switch ID, Port Number
- Packet counts (Received/Sent)
- Byte counts (Received/Sent)
- Port alive duration
- Dropped packets/errors
- Delta features (rate of change)
- Connection Point identifiers
- Load/Rate metrics
- Flow table statistics
- ... (32 total features)

**Note**: The label column should be named `Label` in the CSV (automatically renamed to `Class`)

---

## ⚙️ Configuration

### Modify Classifiers
Edit `Classify.py` to add/remove models or adjust hyperparameters:
```python
classifiers = [
    HistGradientBoostingClassifier(...),
    MLPClassifier(...),
    # Add your classifier here
]
```

### Change Test Splits
Edit `IDS.py` to modify split percentages:
```python
CL.classify(dataset, csv_filename, testSize=20)  # Change 20 to desired %
CL.classify(dataset, csv_filename, testSize=50)  # Change 50 to desired %
```

### Adjust Ensemble Weights
Modify the `weights` list in `Classify.py`:
```python
weights = [0.25, 0.01, 0.02, 0.25, 0.18, 0.09, 0.05]  # Must sum to 1.0
```

---

## 🔬 Performance Metrics

The system calculates both **per-class** and **aggregate** metrics:

- **TPR (True Positive Rate)**: Sensitivity/recall
- **FPR (False Positive Rate)**: Type I error rate
- **Accuracy**: Overall correctness
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value

All metrics are computed per-class then averaged for final comparison.

---

## 🛡️ Use Cases

- **Network Security Monitoring**: Real-time IDS deployment
- **Research**: Compare ML algorithms for intrusion detection
- **Academic**: Benchmark dataset analysis
- **Enterprise**: Evaluate detection difficulty of different attack types
- **Model Selection**: Identify best classifier for specific threat types

---

## ⚠️ Important Notes

1. **Minimum Data**: Requires at least 29 records for meaningful analysis
2. **Missing Values**: Automatically handles NaN via linear interpolation
3. **File Naming**: Results files must follow pattern `*.results.txt` for CSV export
4. **Reproducibility**: `random_state=0` ensures consistent results
5. **Class Imbalance**: All classifiers use balanced weighting

---

## 🐛 Troubleshooting

**Issue**: "No results files found"
- **Solution**: Run `IDS.py` first to generate `.results.txt` files

**Issue**: "Cannot overwrite combined_results.csv"
- **Solution**: Close the CSV file if open in Excel/another program

**Issue**: Import errors
- **Solution**: Verify all packages in `requirements.txt` are installed:
  ```bash
  pip install pandas numpy scikit-learn seaborn matplotlib
  ```

**Issue**: Memory errors on large datasets
- **Solution**: Reduce dataset size or decrease `n_estimators` in tree-based models

---

## 📚 Dependencies

```
pandas>=1.5.0      # Data manipulation
numpy>=1.24.0      # Numerical operations
scikit-learn>=1.3  # ML algorithms
matplotlib>=3.7    # Plotting
seaborn>=0.13.2    # Statistical visualizations
```


---

## 📝 License

This project is provided as-is for research and educational purposes.

---

## 🤝 Contributing

- **Abdulaziz Alessa**
    
- **Turki Alsalama**
    
- **Ziyad Abdulqader** 
    
- **Mutlaq Allahaydan**
    
- **Omran Alharbi**

**Instructor**: Shahid Alam
**Date**: 12/10/2025
Licensed under the MIT License

---
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/username/repo.svg?style=social)](https://github.com/Turki-AlTamimi/ML-Based-Intrusion-Detection)
---

**⭐ If this project helps your research, please consider giving it a star!**

For questions or issues, please open a GitHub issue or contact the author.
