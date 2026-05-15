import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

alpha = 0.05
z_alpha = norm.ppf(1 - alpha)  # 1.645
sigma_diff = np.sqrt(2/3 + 1/2)  # 1.080

delta = np.linspace(0, 5, 200)
power = 1 - norm.cdf(z_alpha - delta / sigma_diff)

plt.figure(figsize=(8, 5))
plt.plot(delta, power, 'b-', linewidth=2)
plt.axhline(y=alpha, color='r', linestyle='--', label=f'α = {alpha}')
plt.axvline(x=0, color='gray', linestyle=':')
plt.xlabel('Истинная разность средних δ = a − b')
plt.ylabel('Мощность (1 − β)')
plt.title('График мощности критерия для H₁: a > b\n(известные дисперсии, n=3, m=2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.xlim(0, 5)
plt.show()

# Мощность при наблюдаемой разности
delta_obs = abs(-1.5967 - (-2.60))  # 1.0033
power_obs = 1 - norm.cdf(z_alpha - delta_obs / sigma_diff)
print(f"Мощность при δ = {delta_obs:.3f}: {power_obs:.3f}")
