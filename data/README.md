# Experimental Results

**File:** `results.pkl` (Pandas DataFrame pickle)

## Schema

| Column | Type | Description |
|:---|---:|---|
| `size` | int | Number of samples (500, 2000, 10000, 50000) |
| `dimension` | int | Number of features (50) |
| `label_noise` | int | Label noise % (0, 10, 20, 30) |
| `feature_noise` | int | Feature noise % (0, 30, 60) |
| `svm_acc` | float | SVM accuracy (average over 5 seeds) |
| `lr_acc` | float | Logistic Regression accuracy |
| `rf_acc` | float | Random Forest accuracy |
| `svm_f1` | float | SVM F1-score |
| `lr_f1` | float | Logistic Regression F1-score |
| `rf_f1` | float | Random Forest F1-score |

## Configurations

48 rows total = 4 sizes x 4 label noises x 3 feature noises.
Each configuration was evaluated 5 times with different random seeds.
