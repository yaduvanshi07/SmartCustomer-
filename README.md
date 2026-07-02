# SmartCustomer — Customer Segmentation using Unsupervised Learning

An unsupervised machine learning project that segments retail customers into distinct groups based on their demographics, spending behavior, and purchasing patterns. The pipeline combines feature engineering, dimensionality reduction (PCA), and clustering (K-Means and Agglomerative Clustering) to uncover actionable customer personas.

## Overview

SmartCustomer analyzes a customer dataset to answer a core marketing question: *what natural groups exist among our customers, and how do they differ?* Rather than predicting a known label, the project discovers hidden structure in the data by:

- Engineering meaningful features from raw demographic and transactional fields
- Reducing dimensionality with PCA for visualization and noise reduction
- Determining the optimal number of clusters using the Elbow Method (via `kneed`)
- Segmenting customers with Agglomerative Clustering
- Profiling each resulting segment by income, spending, and engagement metrics

## Dataset

The notebook expects a CSV file named `smartcart_customers.csv` in the project root. The raw dataset contains 2,240 customer records across 22 columns, including:

| Category | Fields |
|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Enrollment | `Dt_Customer`, `Recency` |
| Spending (last 2 years) | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase behavior | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Engagement | `Complain`, `Response` |

> **Note:** The dataset file is not included in this repository. Place `smartcart_customers.csv` in the project root before running the notebook.

## Project Workflow

### 1. Data Cleaning & Feature Engineering
- Missing `Income` values imputed with the median
- `Age` derived from `Year_Birth` (relative to 2026)
- `Customer_ten` (customer tenure in days) derived from `Dt_Customer` relative to the most recent enrollment date
- `Tot_spending` engineered as the sum of all six spending categories
- `Tot_child` engineered as the sum of `Kidhome` and `Teenhome`
- `Education` regrouped into three tiers: Undergraduate, Graduate, Postgraduate
- `Marital_Status` regrouped into `Living_With`: Partner or Alone

### 2. Outlier Removal & Cleanup
- Dropped redundant/raw columns (`ID`, `Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and the six individual spending columns) after engineering their replacements
- Removed outliers: customers older than 90 and incomes above ₹600,000
- Final cleaned dataset: 2,236 rows × 18 columns

### 3. Encoding & Scaling
- `Education` and `Living_With` one-hot encoded
- All features standardized using `StandardScaler`

### 4. Dimensionality Reduction
- PCA reduced to 3 components, together explaining roughly **45%** of total variance (≈23.2% / 11.4% / 10.4% per component)
- 3D scatter plot used to visualize customer distribution in the reduced space

### 5. Clustering
- **Elbow Method**: WCSS computed for k = 1 to 10, with the optimal k identified programmatically using `KneeLocator`, arriving at **k = 4**
- **Agglomerative Clustering** (ward linkage) applied to the PCA-reduced data to produce 4 final customer segments
- Segments visualized in 3D PCA space and via an Income vs. Total Spending scatter plot

### 6. Cluster Profiling
Each of the 4 clusters was summarized by mean values across income, recency, purchase channels, engagement, age, tenure, total spending, and family size, allowing each segment to be interpreted as a distinct customer persona (e.g., high-income/high-spending frequent shoppers vs. lower-income/lower-engagement customers).

## Tech Stack

- **Python**
- **pandas** — data manipulation
- **seaborn / matplotlib** — 2D and 3D visualization
- **scikit-learn**
  - `OneHotEncoder`, `StandardScaler`
  - `PCA`
  - `KMeans`, `AgglomerativeClustering`
- **kneed** — automatic elbow-point detection for optimal cluster count

## Getting Started

### Prerequisites
```bash
pip install pandas seaborn matplotlib scikit-learn kneed jupyter
```

### Running the Notebook
1. Clone the repository
   ```bash
   git clone https://github.com/yaduvanshi07/SmartCustomer-.git
   cd SmartCustomer-
   ```
2. Add `smartcart_customers.csv` to the project root
3. Launch Jupyter and run `Mini.ipynb`
   ```bash
   jupyter notebook Mini.ipynb
   ```

## Repository Structure
```
SmartCustomer-/
├── Mini.ipynb      # Main notebook: feature engineering, PCA, and clustering
└── README.md
```



