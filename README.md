# rfm-segmentation-evaluation

**RFM Customer Segmentation – Reproducible Evaluation**

`evaluate_segmentation.py` reproduces every number reported in the RFM / K-Means / Random Forest
customer segmentation article. It builds RFM features (Recency, Frequency, Monetary) for each customer,
segments customers with K-Means, and checks how well a Random Forest predicts those segments.

The script produces:

1. **Configuration**: seeds, hyperparameters and package versions
2. **Leak-free evaluation**: the scaler, K-Means and the Random Forest are fitted on training customers only and tested on unseen customers. The original (leaky) setup is also run for comparison, along with 5-fold cross-validation.
3. **Cluster validation for K = 2–8**: silhouette, Davies–Bouldin and Calinski–Harabasz scores
4. **Stability**: Adjusted Rand Index (ARI) of K-Means across 20 random seeds and 50 bootstrap resamples
5. **Alternative methods at K = 4**: Gaussian mixture and agglomerative (Ward) clustering
6. **Segment profiles**: mean and median RFM values and customer counts for Champions, Loyal Customers, At Risk and Hibernating / Lost

## Dataset

This project uses the **Online Retail II** dataset, which is publicly available from the UCI Machine Learning Repository:

<https://archive.ics.uci.edu/dataset/502/online+retail+ii>

It contains all transactions of a UK-based online retailer between 01/12/2009 and 09/12/2011, in two
sheets (`Year 2009-2010` and `Year 2010-2011`).

### License

This dataset is licensed under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You may share and adapt the dataset for any purpose, as long as you give appropriate credit.

### Citation (APA 7th edition)

If you use this dataset, please cite it as:

> Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5CG6D

### Cleaning applied

The script applies the same cleaning as the original `train.py`:

- remove rows without a `Customer ID`
- remove cancelled invoices (invoice numbers starting with `C`)
- keep only rows with `Quantity > 0` and `Price > 0`

RFM features are computed per customer, using a snapshot date one day after the last invoice.
Frequency and Monetary are then log-transformed (`log1p`).

## How to use

### 1. Install Python packages

Python 3.9 or newer is recommended.

```bash
pip install pandas numpy scikit-learn openpyxl joblib
```

### 2. Get the data

Download the dataset from the UCI link above, unzip it, and place `online_retail_II.xlsx` in the same
folder as `evaluate_segmentation.py`:

```
research/
├── evaluate_segmentation.py
├── online_retail_II.xlsx
└── README.md
```

### 3. Run the script

With the data file next to the script, no arguments are needed:

```bash
python evaluate_segmentation.py
```

To use a data file stored somewhere else:

```bash
python evaluate_segmentation.py --data path/to/online_retail_II.xlsx
```

To also record the hyperparameters of a Random Forest saved by the original app:

```bash
python evaluate_segmentation.py --model backend/model/artifacts/rf_model.joblib
```

### Options

| Option    | Default                                  | Description                                          |
|-----------|------------------------------------------|------------------------------------------------------|
| `--data`  | `online_retail_II.xlsx` next to the script | Path to the Online Retail II Excel file            |
| `--model` | none                                     | Optional saved Random Forest (`.joblib`) to inspect  |
| `--out`   | `results`                                | Folder where the results JSON is written             |

A full run takes a few minutes, mostly for reading the 45 MB Excel file and running the bootstrap
stability test.

### 4. Read the output

Progress is printed to the console:

```
Customers: 5878
K=2: silhouette=0.419, davies_bouldin=0.889, calinski_harabasz=5792.986
...
Bootstrap ARI by K: {2: 0.963, 3: 0.97, 4: 0.915, ...}
Original (leaky) macro F1: 0.987
Leak-free macro F1:        0.990
Leak-free 5-fold macro F1: 0.990 ± 0.003

Saved results/evaluation_results.json
```

All results are saved to `results/evaluation_results.json`:

| Key                     | Contents                                                       |
|-------------------------|----------------------------------------------------------------|
| `config`                | Seed, K, test size, features, hyperparameters, package versions |
| `original_model_params` | Parameters of the `--model` file (only if given)               |
| `data`                  | Raw rows, cleaned rows, number of customers                    |
| `k_validation`          | Silhouette, Davies–Bouldin, Calinski–Harabasz for K = 2–8      |
| `stability`             | Seed ARI at K = 4; bootstrap ARI (mean, std) for each K        |
| `alternatives_k4`       | Scores and ARI against K-Means for GMM and Ward clustering     |
| `segment_profiles`      | Mean/median RFM values and customer counts per segment         |
| `eval_original_leaky`   | Accuracy, macro F1, classification report, confusion matrix    |
| `eval_leak_free`        | Same metrics plus train-vs-full ARI and feature importances    |
| `eval_leak_free_cv5`    | 5-fold cross-validated accuracy and macro F1 (mean, std)       |

## Reproducibility

All randomness is fixed with `SEED = 42`. Settings such as K, test size, number of seeds and number of
bootstrap resamples are constants at the top of `evaluate_segmentation.py`. Package versions are
recorded in the output JSON, so results can be compared across machines.
