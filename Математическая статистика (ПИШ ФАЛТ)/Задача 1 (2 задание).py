import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fix random seed for reproducibility
np.random.seed(42)

# ================== GENERATE DATA ==================
n = 50
xi = np.random.uniform(-1, 1, size=(n, 5))  # ξ1..ξ5 ~ R(-1,1)

# Define true parameters
beta_true = np.array([3, -2, 1, 1, -1])  # coefficients for ξ1..ξ5
intercept_true = 2

# Generate η
eta = np.zeros(n)
for i in range(n):
    mean_eta = intercept_true + np.dot(xi[i], beta_true)
    eta[i] = np.random.normal(mean_eta, 1.5)

print(f"Generated {n} samples")
print(f"First 5 samples of xi:\n{xi[:5]}")
print(f"First 5 samples of eta:\n{eta[:5]}\n")

# ================== a) CHECK MULTICOLLINEARITY ==================
def check_multicollinearity(X, names):
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = ["const"] + names
    vif_data["VIF"] = [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]
    print("="*60)
    print("a) Multicollinearity Check (VIF):")
    print(vif_data)
    print("Note: VIF > 10 indicates high multicollinearity\n")
    return vif_data

X_names = [f'ξ{i+1}' for i in range(5)]
check_multicollinearity(xi, X_names)

# ================== b) LINEAR REGRESSION & SIGNIFICANCE ==================
X = sm.add_constant(xi)  # add intercept
model = sm.OLS(eta, X).fit()

print("="*60)
print("b) Linear Regression Results:")
print(model.summary())
print("\n")

# ================== c) R^2 AND ITS SIGNIFICANCE ==================
print("="*60)
print("c) R-squared and its significance:")
print(f"R-squared = {model.rsquared:.4f}")
print(f"Adjusted R-squared = {model.rsquared_adj:.4f}")
f_stat = model.fvalue
f_pvalue = model.f_pvalue
print(f"F-statistic = {f_stat:.4f}, p-value = {f_pvalue:.4e}")
if f_pvalue < 0.05:
    print("R-squared is statistically significant (p < 0.05)")
else:
    print("R-squared is NOT statistically significant")
print("\n")

# ================== d) PREDICTION AT x_k = 0 WITH CI ==================
x_new = np.array([1, 0, 0, 0, 0, 0])  # intercept + 5 zeros
prediction = model.get_prediction(x_new)
pred_summary = prediction.summary_frame(alpha=0.05)

print("="*60)
print("d) Prediction at x_k = 0 (all ξ_k = 0):")
print(f"Predicted value = {pred_summary['mean'][0]:.4f}")
print(f"95% CI: [{pred_summary['mean_ci_lower'][0]:.4f}, {pred_summary['mean_ci_upper'][0]:.4f}]")
print("\n")

# ================== e) INDEPENDENCE OF ERRORS (Durbin-Watson) ==================
print("="*60)
print("e) Independence of errors test (Durbin-Watson):")
dw_stat = sm.stats.durbin_watson(model.resid)
print(f"Durbin-Watson statistic = {dw_stat:.4f}")
print("Interpretation: DW ≈ 2 → no autocorrelation, DW < 1 → positive, DW > 3 → negative")
# Ljung-Box test for autocorrelation
lb_test = acorr_ljungbox(model.resid, lags=[10], return_df=True)
print(f"Ljung-Box test (lag=10): p-value = {lb_test['lb_pvalue'].iloc[0]:.4f}")
if lb_test['lb_pvalue'].iloc[0] > 0.05:
    print("No significant autocorrelation → errors appear independent")
else:
    print("Significant autocorrelation detected")
print("\n")

# ================== f) NORMALITY OF ERRORS ==================
print("="*60)
print("f) Normality of errors test (Shapiro-Wilk):")
shapiro_stat, shapiro_p = stats.shapiro(model.resid)
print(f"Shapiro-Wilk: statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4e}")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(model.resid, 'norm', args=(model.resid.mean(), model.resid.std()))
print(f"Kolmogorov-Smirnov: statistic = {ks_stat:.4f}, p-value = {ks_p:.4e}")

if shapiro_p > 0.05:
    print("Errors are normally distributed (p > 0.05)")
else:
    print("Errors are NOT normally distributed")

# Q-Q plot
fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(model.resid, dist="norm", plot=ax)
ax.set_title("Q-Q Plot of Residuals")
plt.tight_layout()
plt.show()
print("\n")

# ================== g) OUTLIER DETECTION ==================
print("="*60)
print("g) Outlier analysis:")
# Cook's distance
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
# Studentized residuals
student_resid = influence.resid_studentized_internal

outliers = np.where(np.abs(student_resid) > 2)[0]
high_cooks = np.where(cooks_d > 4/n)[0]

print(f"Points with |studentized residual| > 2: {outliers}")
print(f"Points with Cook's distance > 4/n ({4/n:.4f}): {high_cooks}")

# Residual plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")

axes[1].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
axes[1].axhline(y=4/n, color='r', linestyle='--', label=f'4/n = {4/n:.4f}')
axes[1].set_xlabel("Observation index")
axes[1].set_ylabel("Cook's distance")
axes[1].set_title("Cook's Distance")
axes[1].legend()
plt.tight_layout()
plt.show()
print("\n")

# ================== h) CROSS-VALIDATION ==================
print("="*60)
print("h) Cross-validation (5-fold):")
lr = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr, xi, eta, cv=kf, scoring='neg_mean_squared_error')
cv_mse = -cv_scores
print(f"5-fold CV MSE scores: {cv_mse}")
print(f"Mean CV MSE: {cv_mse.mean():.4f} (+/- {cv_mse.std():.4f})")
print("\n")

# ================== i) REPEATED MEASUREMENTS AT ONE POINT ==================
print("="*60)
print("i) Adequacy check with 5 repeated measurements at a fixed point:")
fixed_point = np.array([0, 0, 0, 0, 0])  # all ξ_k = 0
true_mean_at_point = intercept_true + np.dot(fixed_point, beta_true)  # = 2
repeated_eta = [np.random.normal(true_mean_at_point, 1.5) for _ in range(5)]
pred_at_point = model.predict(sm.add_constant(fixed_point.reshape(1, -1)))[0]

print(f"True mean at point: {true_mean_at_point}")
print(f"Predicted mean from model: {pred_at_point:.4f}")
print(f"5 repeated measurements: {repeated_eta}")
print(f"Sample mean of repeats: {np.mean(repeated_eta):.4f}")
print(f"Sample std of repeats: {np.std(repeated_eta):.4f}")

# Compare prediction with repeated measurements
t_stat, p_val = stats.ttest_1samp(repeated_eta, pred_at_point)
print(f"T-test comparing model prediction vs repeats: p-value = {p_val:.4f}")
if p_val > 0.05:
    print("Model is adequate (prediction not significantly different from repeated measurements)")
else:
    print("Model may be inadequate")
print("\n")

# ================== j) REMOVE LEAST SIGNIFICANT VARIABLE ==================
print("="*60)
print("j) Remove least significant variable and re-evaluate:")
pvalues = model.pvalues[1:]  # exclude intercept
least_sig_idx = np.argmax(pvalues)
least_sig_name = X_names[least_sig_idx]
print(f"Least significant variable: {least_sig_name} (p-value = {pvalues[least_sig_idx]:.4f})")

# Remove that variable
X_reduced = np.delete(xi, least_sig_idx, axis=1)
X_reduced_with_const = sm.add_constant(X_reduced)
model_reduced = sm.OLS(eta, X_reduced_with_const).fit()

print(f"\nReduced model results (without {least_sig_name}):")
print(model_reduced.summary())

print(f"\nComparison:")
print(f"Full model R² = {model.rsquared:.4f}, Adjusted R² = {model.rsquared_adj:.4f}")
print(f"Reduced model R² = {model_reduced.rsquared:.4f}, Adjusted R² = {model_reduced.rsquared_adj:.4f}")

# Compare using F-test for nested models
f_test = model.compare_f_test(model_reduced)
print(f"F-test comparing full vs reduced: F = {f_test[0]:.4f}, p-value = {f_test[1]:.4e}")
if f_test[1] < 0.05:
    print("Full model is significantly better than reduced model")
else:
    print("Reduced model is not significantly worse (simpler model preferred)")
print("\n")

# ================== k) BOOTSTRAP COMPARISON ==================
print("="*60)
print("k) Bootstrap comparison of regression equations:")

n_bootstrap = 1000
n_samples = len(eta)
bootstrap_coefs_full = []
bootstrap_coefs_reduced = []

for _ in range(n_bootstrap):
    # Bootstrap sample with replacement
    idx = np.random.choice(n_samples, n_samples, replace=True)
    xi_boot = xi[idx]
    eta_boot = eta[idx]
    
    # Full model
    X_boot_full = sm.add_constant(xi_boot)
    model_boot_full = sm.OLS(eta_boot, X_boot_full).fit()
    bootstrap_coefs_full.append(model_boot_full.params)
    
    # Reduced model
    xi_boot_reduced = np.delete(xi_boot, least_sig_idx, axis=1)
    X_boot_reduced = sm.add_constant(xi_boot_reduced)
    model_boot_reduced = sm.OLS(eta_boot, X_boot_reduced).fit()
    bootstrap_coefs_reduced.append(model_boot_reduced.params)

bootstrap_coefs_full = np.array(bootstrap_coefs_full)
bootstrap_coefs_reduced = np.array(bootstrap_coefs_reduced)

# Calculate bootstrap confidence intervals
print("\nFull model coefficients (bootstrap 95% CI):")
for i, name in enumerate(["const"] + X_names):
    ci_lower = np.percentile(bootstrap_coefs_full[:, i], 2.5)
    ci_upper = np.percentile(bootstrap_coefs_full[:, i], 97.5)
    print(f"  {name}: [{ci_lower:.4f}, {ci_upper:.4f}]")

print("\nReduced model coefficients (bootstrap 95% CI):")
reduced_names = ["const"] + [X_names[j] for j in range(5) if j != least_sig_idx]
for i, name in enumerate(reduced_names):
    ci_lower = np.percentile(bootstrap_coefs_reduced[:, i], 2.5)
    ci_upper = np.percentile(bootstrap_coefs_reduced[:, i], 97.5)
    print(f"  {name}: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Compare R² distributions
bootstrap_r2_full = []
bootstrap_r2_reduced = []
for _ in range(200):  # fewer for speed
    idx = np.random.choice(n_samples, n_samples, replace=True)
    xi_boot = xi[idx]
    eta_boot = eta[idx]
    
    model_full = sm.OLS(eta_boot, sm.add_constant(xi_boot)).fit()
    xi_reduced_boot = np.delete(xi_boot, least_sig_idx, axis=1)
    model_reduced = sm.OLS(eta_boot, sm.add_constant(xi_reduced_boot)).fit()
    bootstrap_r2_full.append(model_full.rsquared)
    bootstrap_r2_reduced.append(model_reduced.rsquared)

print(f"\nBootstrap R² comparison:")
print(f"Full model R²: mean = {np.mean(bootstrap_r2_full):.4f}, std = {np.std(bootstrap_r2_full):.4f}")
print(f"Reduced model R²: mean = {np.mean(bootstrap_r2_reduced):.4f}, std = {np.std(bootstrap_r2_reduced):.4f}")

# Paired bootstrap test for R² difference
diff_r2 = np.array(bootstrap_r2_full) - np.array(bootstrap_r2_reduced)
ci_diff = np.percentile(diff_r2, [2.5, 97.5])
print(f"95% CI for R² difference (full - reduced): [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")
if ci_diff[0] > 0:
    print("Full model has significantly higher R² (bootstrap)")
elif ci_diff[1] < 0:
    print("Reduced model has significantly higher R² (bootstrap)")
else:
    print("No significant difference in R² between models (bootstrap)")

print("\n" + "="*60)
print("All tasks completed!")
