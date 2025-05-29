# NYC Taxi Data Analytics Project

**Authors:** Iachim Cristian & Serbicean Alexandru  
**Date:** May 2025
**Status:** Production Ready  

## Project Overview

This project implements a comprehensive machine learning system for NYC taxi fare prediction and classification. The system provides multiple prediction capabilities through a production-ready REST API.

## Features

- **Binary Classification:** High-value vs. regular fare identification (99.78% accuracy)
- **4-Class Classification:** Fare range segmentation (99.60% accuracy)
- **Regression Analysis:** Exact fare amount prediction (99.51% R² score)
- **Clustering Analysis:** Customer segmentation insights

## Performance Metrics

- **Binary Classification:** 99.78% accuracy (Gradient Boosting)
- **4-Class Classification:** 99.60% accuracy (Gradient Boosting)
- **Regression:** 99.51% R² score (Random Forest)
- **Clustering:** 92% silhouette score (KMeans)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Complete analysis
python main.py

# 4-class classification analysis
python run_multiclass_classification.py

# Model comparison (Phase 5)
python run_phase5_evaluation.py

# Cross-validation analysis
python run_cross_validation.py
```

**Note for Windows users:** If you encounter DLL load errors with TensorFlow, you may need to:
- Install the Visual C++ Redistributable packages
- Use a compatible version of TensorFlow for your Python version
- See the [TensorFlow installation troubleshooting guide](https://www.tensorflow.org/install/errors)

The dashboard will work without TensorFlow, but the Deep Learning model option will be disabled.

## Running the Dashboard

## Project Structure

```
├── src/                           # Core ML modules
│   ├── multiclass_classification.py
│   ├── model_comparison.py
│   ├── cross_validation.py
│   └── ...
├── data/                          # Dataset
├── outputs/                       # Generated results
├── main.py                        # Main analysis script
├── run_multiclass_classification.py
├── run_phase5_evaluation.py
└── requirements.txt
```

## Documentation

- **Final_Technical_Report.md** - Comprehensive technical documentation
- **Project_Summary_Phase6.md** - Executive project summary

## Production Deployment

The system is production-ready with:

- Docker containerization support
- Kubernetes deployment configurations
- Comprehensive monitoring and alerting
- Model versioning and rollback capabilities
- Security and authentication features

See the technical documentation for detailed deployment instructions.

## License

This project is developed for educational and research purposes. 