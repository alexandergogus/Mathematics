import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# ======================
# 1. Исходные данные
# ======================
n1 = 139          # древние египтяне
n2 = 1000         # европейцы

# Стандартные отклонения
s1_len = 5.722
s1_width = 4.612
s2_len = 6.161
s2_width = 5.055

# Степени свободы
df1 = n2 - 1      # 999 (для большей дисперсии)
df2 = n1 - 1      # 138 (для меньшей дисперсии)

# ======================
# 2. Расчёт F-статистики
# ======================
# Длина: большая дисперсия у европейцев
var2_len = s2_len ** 2
var1_len = s1_len ** 2
F_len = var2_len / var1_len  # ≈ 1.159

# Ширина: большая дисперсия у европейцев
var2_width = s2_width ** 2
var1_width = s1_width ** 2
F_width = var2_width / var1_width  # ≈ 1.201

print("=" * 50)
print("F-статистики:")
print(f"  Длина:  F = {F_len:.4f}")
print(f"  Ширина: F = {F_width:.4f}")
print()

# ======================
# 3. Критические значения
# ======================
alpha = 0.05

# Верхнее критическое (правостороннее)
F_crit_upper = f.ppf(1 - alpha/2, df1, df2)

# Нижнее критическое (левостороннее)
F_crit_lower = f.ppf(alpha/2, df1, df2)

print(f"Критические значения (α={alpha}, двусторонний):")
print(f"  Нижний:  {F_crit_lower:.4f}")
print(f"  Верхний: {F_crit_upper:.4f}")
print()

# ======================
# 4. Вывод о гипотезе
# ======================
def test_decision(F, lower, upper):
    if F < lower or F > upper:
        return "Отвергаем H₀ (различия значимы)"
    else:
        return "Не отвергаем H₀ (различия случайны)"

print("Результаты F-теста:")
print(f"  Длина:  {test_decision(F_len, F_crit_lower, F_crit_upper)}")
print(f"  Ширина: {test_decision(F_width, F_crit_lower, F_crit_upper)}")
print()

# ======================
# 5. P-value (двусторонний)
# ======================
def two_sided_p_value(F, df1, df2):
    if F >= 1:
        p_right = 1 - f.cdf(F, df1, df2)
        return 2 * p_right
    else:
        p_left = f.cdf(F, df1, df2)
        return 2 * p_left

p_len = two_sided_p_value(F_len, df1, df2)
p_width = two_sided_p_value(F_width, df1, df2)

print("Двусторонние p-value:")
print(f"  Длина:  {p_len:.4f}")
print(f"  Ширина: {p_width:.4f}")
print()

# ======================
# 6. График мощности критерия
# ======================
# Истинное отношение стандартных отклонений: theta = sigma2 / sigma1
# Под H₀: theta = 1
# Под H₁: theta != 1

theta_range = np.linspace(0.5, 2.0, 200)
power = []

# Нецентральность F-распределения для двухстороннего теста
# Мощность = P(F > F_upper | theta) + P(F < F_lower | theta)
# F_upper и F_lower взяты при theta=1 (центральное)

for theta in theta_range:
    # Отношение дисперсий lambda = theta^2
    lam = theta ** 2
    
    # При истинном lam статистика F имеет нецентральное распределение
    # с параметром нецентральности ncp = df1 * df2 * (lam - 1)**2 / (df1 + df2) ??? 
    # Упрощённо: F ~ (sigma2^2 / sigma1^2) * F_central
    # Но проще смоделировать мощность через нецентральное F
    
    # Для двустороннего теста:
    # Вероятность попасть в критическую область при данном lam
    # F_crit_upper и F_crit_lower из центрального распределения
    
    if lam >= 1:
        # Масштабируем статистику: F_observed = (s2^2/s1^2) ~ lam * F_central(df1, df2)
        # Критические значения остаются теми же
        # P(F > F_upper) = 1 - P(F_central <= F_upper / lam)
        prob_upper = 1 - f.cdf(F_crit_upper / lam, df1, df2)
        prob_lower = f.cdf(F_crit_lower / lam, df1, df2)
    else:
        # lam < 1: меняем ролями, либо считаем так же
        # F_observed = lam * F_central, но lam < 1
        prob_upper = 1 - f.cdf(F_crit_upper / lam, df1, df2)
        prob_lower = f.cdf(F_crit_lower / lam, df1, df2)
    
    power.append(prob_upper + prob_lower)

# График
plt.figure(figsize=(10, 6))
plt.plot(theta_range, power, 'b-', linewidth=2)
plt.axhline(y=alpha, color='r', linestyle='--', label=f'α = {alpha} (уровень значимости)')
plt.axvline(x=1.0, color='gray', linestyle=':', label='θ = 1 (H₀)')
plt.axvline(x=np.sqrt(F_len), color='green', linestyle='--', alpha=0.7, label=f'θ (длина) = {np.sqrt(F_len):.3f}')
plt.axvline(x=np.sqrt(F_width), color='orange', linestyle='--', alpha=0.7, label=f'θ (ширина) = {np.sqrt(F_width):.3f}')

plt.xlabel('Истинное отношение стандартных отклонений θ = σ₂/σ₁', fontsize=12)
plt.ylabel('Мощность (1 - β)', fontsize=12)
plt.title('График мощности двустороннего F-теста\n(сравнение дисперсий, n₁=139, n₂=1000)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.xlim(0.5, 2.0)

# Добавим аннотацию
plt.annotate(f'Мощность при θ={np.sqrt(F_len):.3f} ≈ {power[ np.argmin(np.abs(theta_range - np.sqrt(F_len))) ]:.2f}',
             xy=(np.sqrt(F_len), power[ np.argmin(np.abs(theta_range - np.sqrt(F_len))) ]),
             xytext=(1.15, 0.3), arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.show()

# ======================
# 7. Дополнительно: мощность при наблюдаемых θ
# ======================
def power_at_theta(theta, df1, df2, F_lower, F_upper, alpha):
    lam = theta ** 2
    if lam >= 1:
        prob_upper = 1 - f.cdf(F_upper / lam, df1, df2)
        prob_lower = f.cdf(F_lower / lam, df1, df2)
    else:
        prob_upper = 1 - f.cdf(F_upper / lam, df1, df2)
        prob_lower = f.cdf(F_lower / lam, df1, df2)
    return prob_upper + prob_lower

theta_len = np.sqrt(F_len)
theta_width = np.sqrt(F_width)

power_len = power_at_theta(theta_len, df1, df2, F_crit_lower, F_crit_upper, alpha)
power_width = power_at_theta(theta_width, df1, df2, F_crit_lower, F_crit_upper, alpha)

print("Мощность критерия при наблюдаемых отношениях:")
print(f"  Длина  (θ={theta_len:.3f}):  {power_len:.3f}")
print(f"  Ширина (θ={theta_width:.3f}): {power_width:.3f}")
print()
print("→ Мощность мала (< 0.3), поэтому даже при истинном различии")
print("  вероятность его обнаружить невелика.")
