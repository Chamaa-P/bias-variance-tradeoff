# Bias-Variance Tradeoff in Linear Regression

**MIE 1624H Introduction to Data Science**  
**Group 8 Project**  
**Presentation Date:** March 3, 2026

## 📊 Project Overview

This project provides a comprehensive exploration of the bias-variance tradeoff in linear regression, with practical demonstrations using real-world student performance data. We implement and compare multiple regression techniques, including OLS Linear Regression and ℓ1-regularized (Lasso) regression using both scikit-learn and CVXPY optimization.

## 📁 Dataset

**[Student Productivity and Behavior Dataset (20k)](https://www.kaggle.com/datasets/algozee/student-productivity-and-behavior-dataset-20k)**

- **Size:** ~20,000 student records
- **Target Variable:** GPA (0-4.0 scale)
- **Features:** 9 numeric features including:
  - `study_hours` - Daily study hours (2-10)
  - `sleep_hours` - Daily sleep hours (4-10)
  - `attendance_rate` - Class attendance percentage (50-100%)
  - `social_media_hours` - Daily social media usage (0-6)
  - `netflix_hours` - Daily Netflix hours (0-4)
  - `extracurricular_hours` - Weekly extracurricular hours (0-10)
  - `exercise_frequency` - Days per week (0-7)
  - `diet_quality` - Self-rated 1-10
  - `mental_health_rating` - Self-rated 1-10

## 🎯 Key Topics Covered

1. **Bias-Variance Tradeoff** - Theoretical explanation and empirical visualization
2. **Hyperparameter Tuning** - GridSearchCV and manual tuning for regularization strength (α)
3. **Cross-Validation** - K-fold CV and its relationship to generalization error
4. **Model Evaluation** - R², MAE, MSE, RMSE metrics
5. **Lasso Regression** - Feature selection via ℓ1 regularization
6. **Direct Optimization** - CVXPY implementation of Lasso problem
7. **Learning Curves** - Diagnostic plots for bias and variance
8. **Polynomial Features** - Demonstrating overfitting with model complexity

## 🔧 Requirements

```python
numpy >= 1.20.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
cvxpy >= 1.2.0
kagglehub
```

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn cvxpy kagglehub
   ```

2. **Open the notebook:**
   ```bash
   jupyter notebook bias_variance_tradeoff.ipynb
   ```

3. **Run all cells** - The dataset will be automatically downloaded via `kagglehub`

## 📈 Key Findings

| Metric | OLS Linear Regression | Lasso (α ≈ 0.001-0.01) |
|--------|----------------------|------------------------|
| **R² Score** | ~0.89-0.90 | ~0.89-0.90 |
| **Feature Selection** | Uses all features | Automatically zeros weak predictors |
| **Generalization** | Good | Slightly better (lower variance) |

### Bias-Variance Insights

- **Low complexity (degree 1-2):** High bias, underfitting
- **Optimal complexity:** Minimizes Bias² + Variance
- **High complexity (degree 8+):** High variance, overfitting (especially with small datasets)

### Regularization Effects

- **High α:** Strong regularization → High bias, low variance
- **Optimal α:** Found via cross-validation (~0.001-0.01)
- **Low α:** Weak regularization → Low bias, high variance

## 🔬 Technical Implementation

### Scikit-learn Models
- `LinearRegression()` - OLS baseline
- `Lasso(alpha=...)` - ℓ1-regularized regression
- `LassoCV()` - Automatic α selection via cross-validation
- `PolynomialFeatures()` - Model complexity control

### CVXPY Direct Optimization
Solves the Lasso problem directly:

$$\min_{\beta} \; \frac{1}{m} \|X\beta - y\|_2^2 + \alpha \|\beta\|_1$$

**Verification:** Scikit-learn and CVXPY produce numerically identical results (coefficient differences < 10⁻³)

## 📊 Visualizations

- **Correlation heatmaps** - Feature relationships
- **Scatter plots with trend lines** - Feature vs GPA
- **Bias-variance decomposition** - Across model complexity and α values
- **Learning curves** - Training set size vs error
- **Regularization paths** - Coefficient shrinkage with α
- **U-shape plots** - Classic overfitting visualization

## 📂 Project Structure

```
bias-variance-tradeoff/
├── bias_variance_tradeoff.ipynb    # Main analysis notebook
└── README.md                        # This file
```

## 👥 Authors

**Group 8**  
University of Toronto - MIE 1624H Introduction to Data Science

## 📝 License

This project is for educational purposes as part of MIE 1624H coursework.

## 🙏 Acknowledgments

- Dataset: [Algozee on Kaggle](https://www.kaggle.com/datasets/algozee/student-productivity-and-behavior-dataset-20k)
- Course: MIE 1624H Introduction to Data Science, University of Toronto
- Libraries: scikit-learn, CVXPY, pandas, NumPy, matplotlib, seaborn
