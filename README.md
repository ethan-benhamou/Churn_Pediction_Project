# Churn Prediction Project

##  Project Overview
This project tackles a **customer churn prediction** problem in the context of a **music streaming service**.  
The objective is to predict which users will cancel their subscription within a **10-day window**, using historical user interaction data.

The project was developed as part of a **machine learning competition**, with **ROC-AUC** as the evaluation metric. Due to severe class imbalance and a distribution shift between train and test sets, the problem was reframed as a **ranking task rather than a standard classification task**.

---

##  Key Challenges
- **Extreme class imbalance** (~3% churners in training data)
- **Artificially balanced test set** (~50% churners)
- **Risk of data leakage** from cancellation-related events
- **Probability calibration not reliable** → ranking quality matters more than raw probabilities

---

##  Methodology

### 1. Problem Reformulation
Instead of predicting calibrated churn probabilities, the task is treated as a **ranking problem**:
> Learn a scoring function that correctly orders users by churn risk.

This directly aligns with the **ROC-AUC** metric.

---

### 2. Feature Engineering (Core Contribution)
Rather than relying on raw counts (biased by user tenure), the project focuses on **rate-based and behavioral features**, capturing *changes* in user behavior.

**Main feature groups:**
- **Engagement intensity**
  - Standard deviation of song length
  - Songs per minute
- **Social investment**
  - Add-friend rate
  - Playlist addition rate
- **Frustration signals**
  - Advertisement rate
  - Thumbs-down rate
  - Error rate
- **Temporal dynamics**
  - Days since last activity
  - Activity trends (7-day / 14-day ratios)
- **Satisfaction metrics**
  - Thumbs-up / thumbs-down ratio
  - Unique artists ratio
- **Composite risk score**
  - Domain-driven heuristic combining inactivity, negative feedback, and declining trends

> The **most predictive feature** turned out to be the **standard deviation of song length**, revealing that listening variability is a strong engagement signal.

---

### 3. Model Training Strategy
- **Multi-window training** using three temporal cutoffs  
  → Triples training data without synthetic oversampling
- **Models tested**
  - Logistic Regression
  - LDA
  - LightGBM
  - CatBoost
  - XGBoost
- **Hyperparameter optimization**
  - Optuna
  - 5-fold stratified cross-validation
  - Strong regularization and low learning rates

---

### 4. Ensembling
Instead of averaging probabilities, the final solution uses **rank averaging**:
1. Convert each model’s predictions into ranks
2. Average ranks across models

This approach is:
- Robust to calibration differences
- Perfectly aligned with ROC-AUC optimization

---

##  Results
- **Cross-validation AUC:** ~0.75 – 0.77  
- **Best private leaderboard AUC:** **0.65198**
- **Final submission AUC:** **0.64308**

Despite the extreme imbalance, the model achieves:
- **~3× improvement over random baseline** in Average Precision
- Stable generalization across models

---

##  What Didn’t Work
- SMOTE oversampling (blurred decision boundaries)
- Pseudo-labeling (introduced bias)
- Aggressive feature dropping based solely on correlation

---

##  Conclusion
This project highlights that **problem formulation and feature engineering matter more than model complexity**.

Key takeaways:
- Reframing churn prediction as a **ranking problem** is crucial
- **Rate-based behavioral features** outperform raw activity counts
- Conservative, well-regularized models generalize best under imbalance

Future extensions could include:
- Time-series modeling
- User segmentation and clustering
- Explicit interaction features

---

##  Tech Stack
- Python
- Jupyter Notebook
- LightGBM, CatBoost, XGBoost
- Scikit-learn
- Optuna
- Pandas / NumPy

---

##  Author
**Ethan Ben Hamou**

If you have questions or would like to discuss the methodology, feel free to reach out.
