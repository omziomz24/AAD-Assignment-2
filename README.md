# ACTL30008 Assignment 2: Housing Price Prediction

## Project Overview

This project focuses on predicting median housing prices using the "Assignt2_data.csv" dataset. The analysis is divided into three main parts:

1.  Comparing the performance of a non-parametric regression model (K-Nearest Neighbors) against a non-linear model (Generalized Additive Model).
2.  Addressing the issue of data censoring in the `medianHouseValue` variable by building classification models.
3.  Developing a hybrid approach that combines the best regression and classification models to improve prediction accuracy.

All models were trained on an 80% split of the data and evaluated on the remaining 20% test set.

---

### Part 1: Regression Modelling (Nonlinear vs. Non-parametric)

The primary goal of this section was to compare the predictive power of K-Nearest Neighbors (KNN) and a Generalized Additive Model (GAM) for estimating house values.

**Methods Used:**

* **Baseline Models:** Initial versions of both KNN and GAM were trained on the original predictors.
* **Model Improvement & Feature Engineering:**
    * New predictive features were created, including:
        * `bedroomsPerRoom` and `incomePerRoom` to capture density and affordability.
        * Distances and directions to major cities (Los Angeles and San Francisco) to model spatial effects.
        * A composite `cityProximityScore`.
    * The categorical `oceanProximity` variable was one-hot encoded for the KNN model.
* **Hyperparameter Tuning:** 5-fold cross-validation was used to select the optimal value of `k` for the KNN models.

**Outcome:**

* The **improved KNN model (with k=7)** significantly outperformed the GAM, achieving the lowest test Mean Squared Error (MSE) of **3.06 billion**.
* The feature engineering steps successfully reduced the test MSE for both models.

---

### Part 2: Classification for Data Censoring

This section addressed the observation that the `medianHouseValue` was capped at $500,001. A binary `censoring` variable was created to identify these instances.

**Methods Used:**

* **Classifier Selection:** Logistic Regression and K-Nearest Neighbors (with cross-validation for `k`) were chosen to predict whether a house price was censored.
* **Evaluation:** The models were compared based on their test error rates.

**Outcome:**

* **Logistic Regression** (using the improved feature set) emerged as the better classifier, achieving a slightly lower test error rate of **2.9%**.

---

### Part 3: A Hybrid Approach

This final part combined the best models from the previous sections to create a procedure aimed at correcting for the censoring bias.

**Method Used:**

1.  House prices were predicted using the best-performing regression model (the **improved KNN**).
2.  The best-performing classifier (**Logistic Regression**) was used to predict if an observation was censored.
3.  For any observation predicted to be censored, the KNN-predicted house price was replaced with the ceiling value of $500,001.

**Outcome:**

* The hybrid model produced a test MSE of **3.07 billion**.
* This was marginally *higher* than the test MSE from the improved KNN model alone. This suggests that for this specific dataset and data split, the hard-clipping of predicted values based on the classifier's output did not improve overall prediction accuracy compared to the more nuanced estimates of the standalone KNN model.
