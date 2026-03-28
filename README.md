# Simulation Study: Double Machine Learning Performance Evaluation

## A. Data Generation Procedure

We generate the collected data $\{X_i, \min (T_i, C_i), \Delta_i\}_{i=1}^n$ through the following procedure. 

1.  **Survival Time $T_i$**: The survival time is generated as:
    $\log T_i = X_i^{\top} \beta_0 + \epsilon_i,$
    where $\epsilon_i \sim \mathcal{N}(0, 0.5^2)$ are standard normal errors independent of $X_i$. 
2.  **Covariates**: We consider a situation where the dimension of covariates is fixed as $p=4$, and the vector $X_i = (X_{i1}, \dots, X_{i4})^\top$. Each predictor is drawn independently from a standard uniform distribution: $X_{ij} \sim \mathcal{U}(0, 1)$ for $j=1, \dots, 4$.
3.  **True Parameters**: We specify the true parameter vector as:
    $\beta_0 = (-0.5, 0.5, 0.5, -0.5)^{\top}$.
4.  **Censoring Mechanism**: Designed to satisfy the conditional independent assumption. Specifically, the individualized censoring time $C_i$ satisfies:
    $\log C_i = X_i^{\top} \gamma_0 + \epsilon'_i$,
    where $\gamma_0 = (-0.5, 0, 2, 0)^{\top}$, and the error $\epsilon'_i \sim \mathcal{N}(0, 2^2)$ is independent of covariates. 
5.  **Indicators**: The censoring indicator is $\Delta_i = I(T_i \le C_i)$. We achieve a censoring rate of approximately **40%**.
6.  **Sample Size**: We set the sample size $n = 500, 1000$.

---

## B. Robustness and Sensitivity Evaluation

Recall that the proposed estimation and inference method for $\beta_0$ involves two nuisance functions:
-  $G_0(y,x) := P (\Delta= 1 \mid T = y, X = x)$
-  $f_0(x) := E[\log T \mid X=x]$

We apply the following alternative estimators for $G_0$ and $f_0$:

### 1. Estimators for $G_0$ (4 Approaches)
* **Kaplan-Meier estimation (`KM`)**: A classical nonparametric method for estimating the unconditional survival function.
* **Cox proportional hazards model (`CPH`)**: A standard semiparametric method for survival analysis.
* **Parametric AFT models (`Weibull`/`Normal`)**: Accelerated failure time models with specified error distributions (Weibull and Normal).

### 2. Estimators for $f_0$ (5 Methods)
Estimations are derived via maximum likelihood estimation under right-censoring, where $\log T = f_0(X) + \epsilon$:
* **XGBoost (XGB)**: A popular machine learning algorithm based on gradient boosting decision trees.
    * `XGB_normal`: $\epsilon$ follows a standard normal distribution.
    * `XGB_logistic`: $\epsilon$ follows a standard logistic distribution.
    * `XGB_extreme`: $\epsilon$ follows a Gumbel distribution.
* **Linear Specification**:
    * `Normal`: $\epsilon$ follows a standard normal distribution.
    * `Weibull`: $\epsilon$ follows a Weibull distribution.

In alignment with the **Double Machine Learning (CF-1)** framework, we implement a **2-fold cross-fitting** process across all combinations of the above working models.

---

## C. High-noise Settings

All settings are identical to Part A, with the exception that we set **$\epsilon \sim \mathcal{N}(0, 1)$** and **$\epsilon' \sim \mathcal{N}(0, 4^2)$** to simulate a high-noise setting.

Tables 3 and 4 present the simulation results for $n=500$ and $1000$. While the proposed method is negatively affected by the increased variance of error terms, several combinations of nuisance estimators still provide consistent and asymptotically normal final estimators with coverage probabilities near 95%. Overall, these results suggest that although high-noise and small-sample situations bring significant challenges, the proposed method maintains reliable performance.

---

## D. Results

To quantify the performance of the proposed method, we report 4 metrics calculated under **200 replications**:
- **Bias**: Average bias
- **SSD**: Sample standard deviation
- **ESE**: Estimated standard error
- **CP**: 95% coverage probability of the final estimator $\hat{\beta}_n$

Four Tables summmarize the simulation results.
The Tables named 'n=500', $n=1000$ show the results under sample size $500$ and $1000$, respectively.
The Tables named 'high-noise_n=500' and 'high-noise_n=1000' display the corresponding results under high-noise settings.
