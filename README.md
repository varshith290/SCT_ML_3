## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/cats-dogs-svm.git
cd cats-dogs-svm
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn pillow
```

### 3. Download the dataset
- Go to [kaggle.com/competitions/dogs-vs-cats/data](https://www.kaggle.com/competitions/dogs-vs-cats/data)
- Click **Join Competition** → **Data** tab → download `train.zip`
- Unzip it — you get a `train/` folder with 25,000 images
- Place the `train/` folder in the same directory as the script

### 4. Run the script
```bash
python cats_dogs_svm.py
```

## What the Script Does

| Step | Description |
|------|-------------|
| 1 | Load and resize all images to 64x64 RGB |
| 2 | Data cleaning — check nulls, validate pixel values, check class balance |
| 3 | Feature engineering — normalize pixels, StandardScaler, PCA (12288 → 150 features) |
| 4 | EDA — pixel brightness, RGB channel analysis, PCA scatter |
| 5 | Train 3 SVM models — Linear, RBF, Polynomial kernels |
| 6 | Evaluate — Accuracy, AUC, Classification Report |
| 7 | Hyperparameter tuning — GridSearchCV on RBF kernel |
| 8 | Visualize — 8 plots saved as PNG |
| 9 | Predict a single custom image |

## Why PCA Before SVM

Each image is 64x64 RGB = **12,288 features**. SVM struggles with this many features because:
- Training time grows exponentially with features
- High-dimensional data causes the curse of dimensionality
- Most pixel features are redundant or noisy

PCA reduces 12,288 features → 150 components while keeping ~95% of variance, making SVM fast and accurate.
