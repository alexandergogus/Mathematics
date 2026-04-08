import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==============================
# Параметры задачи
# ==============================
np.random.seed(123)   # для воспроизводимости
theta_true = 2.0
n = 100
confidence = 0.95
alpha = 1 - confidence
z_alpha_half = stats.norm.ppf(1 - alpha/2)   # 1.96

# ==============================
# d) Генерация выборки объёма n = 100
# ==============================
# Парето с x_min = 1: X = 1 / U^(1/(theta-1))
U = np.random.uniform(0, 1, n)
X = 1 / U ** (1 / (theta_true - 1))   # X ~ Pareto(theta_true)

# Вычисление логарифмов
logX = np.log(X)
S = np.sum(logX)

# ММП оценка theta
theta_hat = 1 + n / S
print(f"=== d) Генерация выборки ===")
print(f"Истинный theta: {theta_true}")
print(f"ММП оценка theta_hat: {theta_hat:.4f}")
print()

# ==============================
# b) Точный доверительный интервал для медианы
# ==============================
chi2_low = stats.chi2.ppf(alpha/2, df=2*n)
chi2_high = stats.chi2.ppf(1 - alpha/2, df=2*n)

median_lower = 2 ** (2*S / chi2_high)
median_upper = 2 ** (2*S / chi2_low)

print(f"=== b) Точный интервал для медианы ===")
print(f"95% ДИ для медианы: [{median_lower:.4f}, {median_upper:.4f}]")
print()

# ==============================
# c) Асимптотический доверительный интервал для theta
# ==============================
se_theta = (theta_hat - 1) / np.sqrt(n)
theta_ci_lower = theta_hat - z_alpha_half * se_theta
theta_ci_upper = theta_hat + z_alpha_half * se_theta

print(f"=== c) Асимптотический интервал для theta ===")
print(f"95% ДИ для theta: [{theta_ci_lower:.4f}, {theta_ci_upper:.4f}]")
print()

# ==============================
# e) Параметрический бутстрап
# ==============================
B = 1000
theta_boot_par = []

for _ in range(B):
    # Генерация из Pareto(theta_hat)
    U_boot = np.random.uniform(0, 1, n)
    X_boot = 1 / U_boot ** (1 / (theta_hat - 1))
    logX_boot = np.log(X_boot)
    S_boot = np.sum(logX_boot)
    theta_boot_par.append(1 + n / S_boot)

theta_boot_par = np.array(theta_boot_par)
ci_par_lower = np.percentile(theta_boot_par, 100*alpha/2)
ci_par_upper = np.percentile(theta_boot_par, 100*(1 - alpha/2))

print(f"=== e) Параметрический бутстрап (theta) ===")
print(f"95% ДИ: [{ci_par_lower:.4f}, {ci_par_upper:.4f}]")
print()

# ==============================
# e) Непараметрический бутстрап
# ==============================
theta_boot_np = []

for _ in range(B):
    X_boot = np.random.choice(X, size=n, replace=True)
    logX_boot = np.log(X_boot)
    S_boot = np.sum(logX_boot)
    theta_boot_np.append(1 + n / S_boot)

theta_boot_np = np.array(theta_boot_np)
ci_np_lower = np.percentile(theta_boot_np, 100*alpha/2)
ci_np_upper = np.percentile(theta_boot_np, 100*(1 - alpha/2))

print(f"=== e) Непараметрический бутстрап (theta) ===")
print(f"95% ДИ: [{ci_np_lower:.4f}, {ci_np_upper:.4f}]")
print()

# ==============================
# f) Сравнение всех интервалов
# ==============================
print(f"=== f) Сравнение интервалов (95%) ===")
print(f"Медиана (точный):        [{median_lower:.4f}, {median_upper:.4f}]   длина = {median_upper - median_lower:.4f}")
print(f"Theta (асимптотический): [{theta_ci_lower:.4f}, {theta_ci_upper:.4f}]   длина = {theta_ci_upper - theta_ci_lower:.4f}")
print(f"Theta (парам. бутстрап): [{ci_par_lower:.4f}, {ci_par_upper:.4f}]   длина = {ci_par_upper - ci_par_lower:.4f}")
print(f"Theta (непарам. бутстрап):[{ci_np_lower:.4f}, {ci_np_upper:.4f}]   длина = {ci_np_upper - ci_np_lower:.4f}")

# Дополнительно: визуализация (гистограммы бутстраповских оценок)
plt.figure(figsize=(10, 4))
plt.hist(theta_boot_par, bins=30, alpha=0.6, label='Параметрический', density=True)
plt.hist(theta_boot_np, bins=30, alpha=0.6, label='Непараметрический', density=True)
plt.axvline(theta_hat, color='red', linestyle='--', label=f'ММП = {theta_hat:.3f}')
plt.axvline(theta_true, color='black', linestyle='-', label=f'Истинный θ = {theta_true}')
plt.xlabel('θ')
plt.ylabel('Плотность')
plt.title('Бутстраповские распределения оценки θ')
plt.legend()
plt.show()