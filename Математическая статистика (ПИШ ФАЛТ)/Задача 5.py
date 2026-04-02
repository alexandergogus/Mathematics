import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# --- Фиксируем параметры ---
np.random.seed(42)  # для воспроизводимости
theta_true = 2.0
n = 100
alpha = 0.05
B = 1000  # число бутстрап-выборок

# --- f) Генерация выборки ---
# X ~ Uniform[theta, 2*theta]
X = uniform.rvs(loc=theta_true, scale=theta_true, size=n)
X_min = np.min(X)
X_max = np.max(X)
X_mean = np.mean(X)

print("=== f) Исходные данные ===")
print(f"Истинное θ = {theta_true}")
print(f"Объём выборки n = {n}")
print(f"X_min = {X_min:.4f}")
print(f"X_max = {X_max:.4f}")
print(f"X_mean = {X_mean:.4f}\n")

# --- Оценки параметра ---
theta_MM = (2/3) * X_mean
theta_ML_crude = X_max / 2
theta_ML_unbiased = ((n+1) / (2*n + 1)) * X_max

print("=== Оценки θ ===")
print(f"Метод моментов (MM):        {theta_MM:.6f}")
print(f"МП (смещённая):             {theta_ML_crude:.6f}")
print(f"МП (исправленная):          {theta_ML_unbiased:.6f}\n")

# --- d) Точный доверительный интервал ---
# Используем статистику X_(n)
q_alpha2 = 1 + (alpha/2)**(1/n)
q_1minus_alpha2 = 1 + (1 - alpha/2)**(1/n)
ci_exact_lower = X_max / q_1minus_alpha2
ci_exact_upper = X_max / q_alpha2

print("=== d) Точный ДИ ===")
print(f"Нижняя граница: {ci_exact_lower:.6f}")
print(f"Верхняя граница: {ci_exact_upper:.6f}")
print(f"Ширина: {ci_exact_upper - ci_exact_lower:.6f}")
print(f"Покрывает истинное θ? {ci_exact_lower <= theta_true <= ci_exact_upper}\n")

# --- e) Асимптотический ДИ (на основе θ_MM) ---
z_crit = norm.ppf(1 - alpha/2)
se_asympt = theta_MM / np.sqrt(27 * n)
ci_asympt_lower = theta_MM - z_crit * se_asympt
ci_asympt_upper = theta_MM + z_crit * se_asympt

print("=== e) Асимптотический ДИ ===")
print(f"Нижняя граница: {ci_asympt_lower:.6f}")
print(f"Верхняя граница: {ci_asympt_upper:.6f}")
print(f"Ширина: {ci_asympt_upper - ci_asympt_lower:.6f}")
print(f"Покрывает истинное θ? {ci_asympt_lower <= theta_true <= ci_asympt_upper}\n")

# --- g) Бутстраповский ДИ (на основе исправленной МП оценки) ---
theta_bootstrap = np.zeros(B)

for i in range(B):
    # Генерируем псевдовыборку с возвращением
    X_boot = np.random.choice(X, size=n, replace=True)
    X_max_boot = np.max(X_boot)
    # Используем исправленную оценку МП
    theta_bootstrap[i] = ((n+1) / (2*n + 1)) * X_max_boot

# Процентный бутстрап-интервал
ci_boot_lower = np.percentile(theta_bootstrap, 100 * alpha/2)
ci_boot_upper = np.percentile(theta_bootstrap, 100 * (1 - alpha/2))

print("=== g) Бутстраповский ДИ (исправленная МП оценка) ===")
print(f"Нижняя граница: {ci_boot_lower:.6f}")
print(f"Верхняя граница: {ci_boot_upper:.6f}")
print(f"Ширина: {ci_boot_upper - ci_boot_lower:.6f}")
print(f"Покрывает истинное θ? {ci_boot_lower <= theta_true <= ci_boot_upper}\n")

# --- h) Сравнение всех интервалов ---
print("=== h) Сравнение доверительных интервалов (уровень 0.95) ===")
print(f"{'Метод':<20} {'Нижняя граница':<15} {'Верхняя граница':<15} {'Ширина':<10} {'Покрытие':<10}")
print("-" * 70)
print(f"{'Точный (на основе X_max)':<20} {ci_exact_lower:<15.6f} {ci_exact_upper:<15.6f} {ci_exact_upper - ci_exact_lower:<10.6f} {'Да' if ci_exact_lower <= theta_true <= ci_exact_upper else 'Нет':<10}")
print(f"{'Асимптотический (θ_MM)':<20} {ci_asympt_lower:<15.6f} {ci_asympt_upper:<15.6f} {ci_asympt_upper - ci_asympt_lower:<10.6f} {'Да' if ci_asympt_lower <= theta_true <= ci_asympt_upper else 'Нет':<10}")
print(f"{'Бутстраповский (МП испр)':<20} {ci_boot_lower:<15.6f} {ci_boot_upper:<15.6f} {ci_boot_upper - ci_boot_lower:<10.6f} {'Да' if ci_boot_lower <= theta_true <= ci_boot_upper else 'Нет':<10}")

# --- Дополнительно: визуализация бутстрап-распределения ---
plt.figure(figsize=(10, 5))
plt.hist(theta_bootstrap, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(theta_true, color='red', linestyle='--', label=f'Истинное θ = {theta_true}')
plt.axvline(ci_boot_lower, color='green', linestyle='--', label=f'2.5% квантиль = {ci_boot_lower:.3f}')
plt.axvline(ci_boot_upper, color='green', linestyle='--', label=f'97.5% квантиль = {ci_boot_upper:.3f}')
plt.xlabel('θ')
plt.ylabel('Плотность')
plt.title('Бутстраповское распределение исправленной МП оценки θ')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- Проверка покрытия для асимптотического интервала (дополнительно, через моделирование) ---
# Покажем, что асимптотический интервал работает плохо при малых n
# но при n=100 уже неплохо
n_sim = 500
cover_asympt = 0
for _ in range(n_sim):
    X_sim = uniform.rvs(loc=theta_true, scale=theta_true, size=n)
    theta_MM_sim = (2/3) * np.mean(X_sim)
    se_asympt_sim = theta_MM_sim / np.sqrt(27 * n)
    lower = theta_MM_sim - z_crit * se_asympt_sim
    upper = theta_MM_sim + z_crit * se_asympt_sim
    if lower <= theta_true <= upper:
        cover_asympt += 1
print(f"\n(Доп.) Эмпирическое покрытие асимптотического ДИ: {cover_asympt/n_sim:.3f} (должно быть ~0.95 при n→∞)")
