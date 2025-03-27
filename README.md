# Cybersecurity Threat Classification using Machine Learning

## Project Overview
A machine learning system to classify network threats using the UNSW-NB15 dataset. Implements Random Forest, SVM, and Neural Network models for intrusion detection.

## Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

## Directory Structure
```
cybersecurity-threat-classification/
├── data/ # Dataset files
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── models/ # Saved models
│   ├── best_cyber_threat_model.pkl
│   ├── scaler.pkl
│   └── selected_features.pkl
├── notebooks/ # Jupyter notebooks
│   └── CyberThreatClassification.ipynb
└── reports/ # Output visualizations
```
Report Link : https://docs.google.com/document/d/18nLbunGatkH1mVYDgD0dYyM-Om16ptOMs_qcoebpYDo/edit?usp=sharing
dataSet Link : https://research.unsw.edu.au/projects/unsw-nb15-dataset
## Installation & Execution
### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/cybersecurity-threat-classification.git
cd cybersecurity-threat-classification
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Download dataset:
- Get UNSW-NB15 dataset
- Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` in `data/` folder

### 4. Run Jupyter Notebook:
```bash
jupyter notebook notebooks/CyberThreatClassification.ipynb
```

## Workflow Steps
### 1. Data Loading
```python
train_df = pd.read_csv('../data/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('../data/UNSW_NB15_testing-set.csv')
df = pd.concat([train_df, test_df])
```

### 2. Data Preprocessing
- Handles missing values with `df.fillna(0)`
- Encodes categorical features using `LabelEncoder()`
- Drops unnecessary columns like `id` and `attack_cat`

### 3. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)
```

### 4. Model Training
```python
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
```

### 5. Model Evaluation
- Generates:
  - Classification reports (precision, recall, f1-score)
  - Confusion matrices
  - Feature importance plots

### 6. Saving Models
```python
import joblib
joblib.dump(best_model, 'models/best_cyber_threat_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
```

## Sample Prediction
```python
loaded_model = joblib.load('models/best_cyber_threat_model.pkl')
sample_data = X_test.iloc[0:1][selected_features]
prediction = loaded_model.predict(scaler.transform(sample_data))
print(f"Predicted: {'Attack' if prediction[0] == 1 else 'Normal'}")
```

## Troubleshooting
| Error | Solution |
|--------|-----------|
| `FileNotFoundError` | Verify CSV files are in `data/` folder |
| `AttributeError: 'numpy.ndarray'...` | Convert arrays to DataFrames with `pd.DataFrame()` |
| Training too slow | Use `.sample(frac=0.1)` for testing |
| Jupyter not found | Install with `pip install jupyter` |

## Expected Outputs
### Console Output:
```
Random Forest Accuracy: 0.94
SVM Accuracy: 0.92
Neural Network Accuracy: 0.93
```

### Visualizations:
- Confusion matrices for each model
- Feature importance plot (Random Forest)

### Saved Models:
- `best_cyber_threat_model.pkl`
- `scaler.pkl`
- `selected_features.pkl`

