# Machine Learning Project: Pokémon Image Classification  
**NOVA School of Science and Technology (NOVA FCT) – 2024/2025**  
**Course:** Machine Learning  
**Final Grade:** 14.3
**Author:** 
  - Ricardo Rodrigues (rf-rodrigues95)
---

## Project Description

https://www.kaggle.com/competitions/machine-learning-nova-2024-the-three-body-proble

This repository contains the code, experiments, and final report for a Machine Learning project focused on predicting the positions of celestial bodies in the Three-Body Problem, purely through data-driven techniques, without using numerical integration or physical simulation.

The project is structured into four main tasks:

1. **Task 1 – Baseline Modeling**
Data exploration, trajectory visualization, and the implementation of a Linear Regression baseline model using initial conditions to predict future positions.

2. **Task 2 – Polynomial Regression & Regularization**
Development and validation of polynomial regression models. Regularization techniques (Ridge, Lasso) are explored to control model complexity. Best-performing models are selected through RMSE analysis and hyperparameter tuning.

3. **Task 3 – Feature Engineering**
Both feature reduction (via correlation analysis) and feature augmentation (e.g., norms, distances, ratios) are performed to improve prediction quality. The impact of these engineered features is evaluated using the polynomial models from Task 2.

4. **Task 4 – Nonparametric Modeling with k-NN**
Implementation and evaluation of a k-Nearest Neighbors Regressor. Experiments investigate how performance and computational cost vary with 
𝑘
k, using both raw and engineered features. Final results are compared against all previous tasks using RMSE and prediction plots.


---

## Repository Structure

- /report/ # Final comprehensive report covering all tasks (PDF)


Each task folder includes all related code and experiments for that stage. The `/report/` folder contains the final report summarizing methodology, results, and insights from all tasks.

---
