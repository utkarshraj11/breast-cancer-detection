# CS235 Methods Project — Machine Learning Techniques

This project was completed as part of **CS235 (Data Mining Techniques)** at UC Riverside. It covers hands-on implementation of a range of machine learning techniques across three parts: data preprocessing, supervised learning, and unsupervised learning. All implementations are in Python using Jupyter Notebooks.

---

## Dataset

The primary dataset used throughout Parts 1 and 2 is the **Wisconsin Breast Cancer Diagnostic dataset**, sourced from Kaggle since the official UCI archive link was returning errors at the time of submission.

- **Features:** 30 numeric features computed from digitized images of cell nuclei (radius, texture, perimeter, etc.)
- **Target:** Binary classification — Malignant (1) vs. Benign (0)
- **Size:** 569 samples, class imbalanced (~63% benign, ~37% malignant)

Part 2 (Transfer Learning) and Part 3 use the **MNIST Handwritten Digits dataset**, split into Dataset A (digits 0–4) and Dataset B (digits 5–9).

---

## Repository Structure

```
├── methods_project_part1.ipynb   # Data Preprocessing
├── methods_project_part2.ipynb   # Supervised Techniques
├── methods_project_part3.ipynb   # Unsupervised Techniques
├── data.csv                      # Wisconsin Breast Cancer dataset
└── top_params_bayes_opt.pkl      # Saved Bayesian optimisation results (used in Part 2 Q2)
```

---

## Part 1 — Data Preprocessing

### 1. Data Cleaning & Missing Value Prediction

The goal here was to understand how well different imputation strategies can recover model performance after introducing missing data. Two types of noise were introduced at p = 20% and p = 40%:

- **Single feature noise** — replace p% of values in the `radius_mean` column with NaN
- **Random feature noise** — for p% of rows, pick a random feature and set it to NaN

Performance was then measured using Logistic Regression under three conditions: no imputation (rows with NaN dropped), `SimpleImputer` (mean imputation), and `IterativeImputer` (multivariate imputation using the relationships between features). All imputers were strictly fitted on training folds only to prevent data leakage. Evaluation was done using stratified 5-fold cross-validation reporting mean ± std F1 score.

### 2. Dimensionality Reduction

Two dimensionality reduction techniques were compared across latent dimensions k = [2, 5, 10]:

- **Truncated SVD** — a linear technique that projects the data onto the top-k singular vectors computed on the training set only. The test set is then projected into the same space (following the Latent Semantic Analysis approach).
- **Autoencoder** — implemented using `MLPRegressor` with an hourglass architecture: input → min(d, 2k) → k (bottleneck) → min(d, 2k) → output, with ReLU activations. The k-dimensional bottleneck output serves as the latent representation. Trained to reconstruct the input, not predict labels.

Latent representations from both methods were fed into Logistic Regression. A baseline (no dimensionality reduction) is plotted as a horizontal reference line.

---

## Part 2 — Supervised Techniques

### 1. Hyperparameter Optimisation

Compared two strategies for tuning a Random Forest classifier:

- **Grid Search** (`GridSearchCV`) — exhaustively evaluates every combination in a defined grid
- **Bayesian Optimisation** (`BayesSearchCV` from scikit-optimize) — uses past evaluations to make smarter guesses about where to look next in the hyperparameter space

Both were tested with a **fine grid** (larger range, more values) and a **coarse grid** (same range, fewer values), giving four combinations in total. Parameters tuned include `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `ccp_alpha`. Each run was averaged over 5 independent iterations to get a reliable time and F1 estimate. Results are plotted as F1 score vs. average time spent.

Best hyperparameters from the Bayesian fine search are saved to `top_params_bayes_opt.pkl` and reused in Q2.

### 2. Data Augmentation (SMOTE)

Explored how SMOTE (Synthetic Minority Over-sampling Technique) can help when the minority class is artificially reduced. The minority (malignant) class was reduced to 25% of its original size using two sampling strategies:

- **Uniform random** — sample 25% of the minority class at random
- **Hardness-based** — fit a Logistic Regression on the full dataset, rank the minority class by P(class=1|data) ascending (lowest confidence = hardest examples), and take the top 25%

SMOTE was then applied with k=1 and k=5 neighbours. A Random Forest (using the Bayes-optimised hyperparameters from Q1) was evaluated with and without SMOTE augmentation, across 5 iterations on a single train/test split. Results are shown as a grouped bar chart comparing all combinations.

### 3. Transfer Learning (CNN)

Trained two CNN architectures on Dataset A (MNIST digits 0–4) and measured how well their learned representations transfer to Dataset B (MNIST digits 5–9):

- **2-Layer CNN** — two Conv2D layers (32 and 64 filters) each followed by MaxPooling2D, then two fully-connected layers
- **3-Layer CNN** — three Conv2D layers (32 → 64 → 64 filters) with pooling after the first two, then two fully-connected layers

For transfer, the convolutional base is frozen and the two FC layers are replaced with freshly initialised ones and trained on varying proportions p% of Dataset B's training data (p ∈ {5, 10, 25, 50, 75, 100}). Performance (macro F1) on a fixed 20% holdout of Dataset B is plotted as a function of p. A benchmark line shows the performance achievable by training each CNN directly on Dataset B from scratch.

---

## Part 3 — Unsupervised Techniques

### Clustering: Raw Data vs. Learned Representations

Used K-Means and Spectral Clustering to cluster the 20% held-out portion of Dataset A (MNIST digits 0–4) that was never seen by the CNN during training. Clustering was applied to two types of input:

- **Raw pixel data** — 28×28 images flattened to 784-dimensional vectors
- **Learned representations** — output of the 2-layer CNN's convolutional base (before the FC layers), after training on the 80% split

## The idea is that the CNN's convolutional base should compress pixel data into a more structured, semantically meaningful feature space, making it easier for clustering algorithms to separate digit classes. Performance is measured using the Silhouette coefficient, averaged over 10 random initialisations (mean ± std) for K = 2 to 5. K=1 is excluded as the silhouette score is undefined for a single cluster.
