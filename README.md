# Ancient Greek Inscription Dating with Neural Networks

A machine learning project that uses neural networks to predict the date of ancient Greek inscriptions based on text analysis.

## 📖 Overview

This project develops a neural network model capable of predicting when ancient Greek inscriptions were written. The model uses text vectorization techniques combined with feedforward neural networks to perform regression on inscription dates.

### Key Features

- **Text Vectorization**: Implements Bag of Words (BoW) model using TF-IDF vectorization
- **Neural Network Architecture**: Customizable feedforward networks with configurable hidden layers
- **Regularization Techniques**: Supports dropout regularization to prevent overfitting
- **Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Early Stopping**: Optional early stopping callback to prevent overfitting

## 🗂️ Project Structure

```
├── code/
│   ├── compute.ipynb    # Main Jupyter notebook with experiments
│   └── utils.py         # Utility functions for model building and training
├── dataset/
│   └── iphi2802.csv     # Dataset of 2,802 ancient Greek inscriptions
├── report/
│   ├── CompIntel1.docx  # Project report (Word format)
│   └── report.pdf       # Project report (PDF format)
└── README.md
```

## 🛠️ Technologies Used

- **Python** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework
- **Scikit-learn** - Machine learning utilities (TF-IDF, K-Fold CV, MinMaxScaler)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **NLTK** - Natural language processing (Greek stopwords)
- **Matplotlib** - Visualization

## 📊 Dataset

The project uses the **IPHI2802** dataset containing:
- **2,802 ancient Greek inscriptions**
- Features include: text, region information, date ranges (min/max), and metadata
- Target variable: Mean date calculated from date_min and date_max

## 🔬 Methodology

### Preprocessing
1. **Text Vectorization**: TF-IDF vectorization with 3,000 features
2. **Stopword Removal**: Ancient Greek stopwords removed using NLTK
3. **Feature Scaling**: MinMaxScaler applied to both input and output data
4. **Cross-Validation Setup**: Dataset split into 5 folds

### Model Architecture
- **Input Layer**: 3,000 nodes (TF-IDF features)
- **Hidden Layers**: Configurable dense layers with ReLU activation
- **Output Layer**: Single node with linear activation (regression)
- **Loss Function**: Custom RMSE (Root Mean Square Error)
- **Optimizer**: Adam with configurable learning rate and momentum

### Training Features
- Configurable batch sizes and epochs
- Dropout regularization for input and hidden layers
- Early stopping with patience parameter
- Training/validation loss tracking per epoch

## 📈 Results

The model demonstrates effective learning with:
- Converging training and validation losses
- RMSE as the primary evaluation metric
- Experiments comparing different architectures and hyperparameters

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow keras scikit-learn pandas numpy nltk matplotlib
```

### Running the Project
1. Open `code/compute.ipynb` in Jupyter Notebook or JupyterLab
2. Run the cells sequentially to:
   - Load and preprocess the dataset
   - Build and train neural network models
   - Evaluate performance across folds
   - Visualize training results

## 📄 License

This project was created as part of a Computational Intelligence course.

---

*Author: Konstantinos*
