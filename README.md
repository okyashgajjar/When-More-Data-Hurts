# When More Data Hurts: An Empirical Study of Noise Sensitivity and Performance Saturation in Machine Learning Models

**Published in:** IJCSPUB (International Journal of Current Science), ISSN: 2250-1770  
**Paper ID:** IJCSP26B1097  
**Volume 16, Issue 2, pp. 676-689, April 2026**  
**Impact Factor:** 8.17 (Calculated by Google Scholar)  
**Link:** [IJCSP26B1097.pdf](https://rjpn.org/IJCSPUB/papers/IJCSP26B1097.pdf)

---

## Abstract

People tend to think that just throwing more data at a machine learning model always improves it. Honestly, that's only part of the story. Data quality matters a lot, too. In this study, we examined how dataset size, label noise, and feature noise affect outcomes using carefully designed synthetic experiments. We put Logistic Regression, SVMs, and Random Forests through their paces, trying out sample sizes from 500 up to 50,000 and testing different levels of noise. Here's what we found: Bigger datasets do help, but only when the data is clean. Once you add label noise, you hit a ceiling—more data doesn't really help all that much. Feature noise is a bit different; if you keep adding more samples, you can offset some of that messiness. We also noticed that more complex models handle noisy data better than simpler ones. Bottom line: if your labels are wrong, pouring in more data doesn't save you. Getting the data right, especially the labels, matters just as much as how much data you have.

---

## Key Findings

### 1. Clean Data Scales Predictably

| Dataset Size | SVM Accuracy | LR Accuracy | RF Accuracy |
|:---:|:---:|:---:|:---:|
| 500 | 0.8672 | 0.7312 | 0.7952 |
| 2,000 | 0.9524 | 0.7036 | 0.8760 |
| 10,000 | 0.9803 | 0.7250 | 0.9450 |
| **50,000** | **0.9888** | **0.7185** | **0.9582** |

With clean data (LN=0%, FN=0%), SVM and RF benefit substantially from larger datasets. LR plateaus early (~0.72).

### 2. Label Noise Causes Performance Saturation

At LN=30%, increasing dataset size from 10,000 to 50,000 yields almost no improvement:

| Dataset Size | SVM (LN=30%) | LR (LN=30%) | RF (LN=30%) |
|:---:|:---:|:---:|:---:|
| 500 | 0.7392 | 0.6560 | 0.6720 |
| 2,000 | 0.7976 | 0.6324 | 0.7424 |
| 10,000 | 0.8325 | 0.6558 | 0.8015 |
| 50,000 | 0.8412 | 0.6516 | 0.8180 |

SVM RF improve only ~1-2% from 10K to 50K. LR fails to improve at all. **More data does not fix bad labels.**

### 3. Feature Noise Can Be Offset With More Data

With FN=60% (no label noise), SVM accuracy recovers as data grows:

| Dataset Size | SVM (FN=60%) |
|:---:|:---:|
| 500 | 0.7744 |
| 2,000 | 0.8684 |
| 10,000 | 0.8645 |
| 50,000 | **0.9235** |

Unlike label noise, feature noise can be compensated with more samples.

### 4. Model Robustness Ranking

| Condition | Best Model | Worst Model |
|:---|:---|:---|
| Clean data | SVM (0.989) | LR (0.719) |
| High label noise (30%) | SVM (0.841) | LR (0.652) |
| High feature noise (60%) | RF (0.944) | LR (0.744) |

SVM dominates on clean data and under label noise. RF handles feature noise best. LR consistently underperforms.

---

## Repository Structure

```
When-More-Data-Hurts/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── CITATION.cff                   # Citation metadata
├── requirements.txt               # Python dependencies
├── .gitignore
├── data/
│   └── results.pkl                # Full experimental results (48 configurations x 10 metrics)
├── notebooks/
│   ├── 01_experiment_setup.ipynb  # Data generation, model training, evaluation pipeline
│   └── 02_analysis.ipynb          # Analysis & visualization (from main.ipynb)
├── figures/
│   ├── accuracy_vs_dataset_size/  # Accuracy/F1 vs dataset size for all noise combos
│   │   ├── clean_scaling/         # Clean data scaling baseline
│   │   └── ln_*_fn_*/             # 12 noise configurations (4 LN x 3 FN levels)
│   ├── noise_effects/             # Label vs feature noise comparison plots
│   │   ├── by_dataset_size/       # 4 dataset sizes x 3 models
│   │   ├── larger_dataset_comparison/   # 50K sample noise comparisons
│   │   └── smaller_dataset_comparison/  # 500 sample noise comparisons
│   ├── model_robustness/          # Bar charts comparing model performance
│   │   ├── all_configs/           # Individual model comparison plots
│   │   ├── large_ds_fn_vary/      # 50K samples, LN=0, FN varies
│   │   ├── large_ds_ln_vary/      # 50K samples, FN=0, LN varies
│   │   ├── small_ds_fn_vary/      # 500 samples, LN=0, FN varies
│   │   └── small_ds_ln_vary/      # 500 samples, FN=0, LN varies
│   ├── fixed_fn_ln_vary/          # FN=0, LN varies (accuracy vs dataset size)
│   │   ├── lr/ / rf/ / svm/
│   └── fixed_ln_fn_vary/          # LN=0, FN varies (accuracy vs dataset size)
│       ├── lr/ / rf/ / svm/
└── paper/
    └── README.md                  # Paper citation and link
```

---

## Experimental Setup

### Synthetic Data
- **Samples:** 500, 2,000, 10,000, 50,000
- **Features:** 50 (50% informative, 50% redundant, + noise features)
- **Class separation:** 0.7
- **Label noise (flip_y):** 0%, 10%, 20%, 30%
- **Feature noise (useless features):** 0%, 30%, 60%

### Models
| Model | Hyperparameters |
|:---|---|
| SVM | RBF kernel, C=1.0, gamma='scale' |
| Logistic Regression | max_iter=2000 |
| Random Forest | 150 estimators |

### Evaluation
- 75/25 stratified train-test split
- 5 random seeds averaged per configuration
- Metrics: Accuracy, F1-score

### Total Configurations
48 (4 sizes x 4 label noises x 3 feature noises), each with 5 repeats = **240 individual evaluations**.

---

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/your-username/When-More-Data-Hurts.git
cd When-More-Data-Hurts

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the experiment (generates results.pkl)
#    Open notebooks/01_experiment_setup.ipynb in Jupyter

# 4. Explore analysis & figures
#    Open notebooks/02_analysis.ipynb
```

---

## Citation

```bibtex
@article{ijcsp26b1097,
  title     = {When More Data Hurts: An Empirical Study of Noise Sensitivity 
               and Performance Saturation in Machine Learning Models},
  author    = {Yash Gajjar, Aarush Jaiminbhai Patel, Sakshi Rajeshbhai Mistry, 
               Dhruvil Sanjaykumar Prajapati, Axita Jitendrakumar Patel, Jalpa Patel},
  journal   = {International Journal of Current Science (IJCSPUB)},
  volume    = {16},
  number    = {2},
  pages     = {676--689},
  year      = {2026},
  issn      = {2250-1770},
  url       = {https://rjpn.org/IJCSPUB/papers/IJCSP26B1097.pdf}
}
```
