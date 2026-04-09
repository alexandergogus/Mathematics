import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample from exponential distribution with rate 1
n = 25
sample = np.random.exponential(scale=1, size=n)

print("Sample:", sample)
print(f"Sample size: {n}")

# a) Sample statistics
sample_mean = np.mean(sample)
sample_median = np.median(sample)

# Mode for continuous distribution - using histogram
hist, bins = np.histogram(sample, bins='auto')
mode_idx = np.argmax(hist)
mode = (bins[mode_idx] + bins[mode_idx + 1]) / 2

# Range
data_range = np.max(sample) - np.min(sample)

# Skewness coefficient
def skewness(x):
    n = len(x)
    m2 = np.sum((x - np.mean(x))**2) / n
    m3 = np.sum((x - np.mean(x))**3) / n
    if m2 == 0:
        return 0
    return m3 / (m2**(3/2))

sample_skewness = skewness(sample)
# Alternatively using scipy
scipy_skewness = stats.skew(sample, bias=False)

print("\na) Sample statistics:")
print(f"Mode: {mode:.4f}")
print(f"Median: {sample_median:.4f}")
print(f"Range: {data_range:.4f}")
print(f"Skewness coefficient: {sample_skewness:.4f}")
print(f"Skewness (scipy, bias corrected): {scipy_skewness:.4f}")

# b) Empirical distribution function, histogram and boxplot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Empirical CDF
sorted_sample = np.sort(sample)
ecdf_y = np.arange(1, n+1) / n
axes[0].step(sorted_sample, ecdf_y, where='post')
axes[0].set_xlabel('x')
axes[0].set_ylabel('F(x)')
axes[0].set_title('Empirical CDF')
axes[0].grid(True, alpha=0.3)

# Histogram
axes[1].hist(sample, bins='auto', density=True, alpha=0.7, edgecolor='black')
x = np.linspace(0, max(sample) + 1, 1000)
axes[1].plot(x, np.exp(-x), 'r-', label='True PDF (exp(-x))')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Density')
axes[1].set_title('Histogram vs True PDF')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Boxplot
axes[2].boxplot(sample)
axes[2].set_ylabel('Value')
axes[2].set_title('Boxplot')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# c) Bootstrap vs CLT for sample mean distribution
B = 1000
bootstrap_means = np.zeros(B)
for i in range(B):
    bootstrap_sample = np.random.choice(sample, size=n, replace=True)
    bootstrap_means[i] = np.mean(bootstrap_sample)

# CLT approximation
clt_mean = sample_mean
clt_std = np.std(sample, ddof=1) / np.sqrt(n)
x_clt = np.linspace(min(bootstrap_means), max(bootstrap_means), 1000)
clt_pdf = stats.norm.pdf(x_clt, clt_mean, clt_std)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=30, density=True, alpha=0.7, 
         edgecolor='black', label='Bootstrap estimate')
plt.plot(x_clt, clt_pdf, 'r-', linewidth=2, label='CLT approximation')
plt.xlabel('Sample mean')
plt.ylabel('Density')
plt.title('Distribution of Sample Mean: Bootstrap vs CLT')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nc) Sample mean distribution comparison:")
print(f"Bootstrap mean: {np.mean(bootstrap_means):.4f}")
print(f"Bootstrap std: {np.std(bootstrap_means):.4f}")
print(f"CLT mean: {clt_mean:.4f}")
print(f"CLT std: {clt_std:.4f}")

# d) Bootstrap for skewness coefficient
B_skew = 1000
bootstrap_skewness = np.zeros(B_skew)
for i in range(B_skew):
    bootstrap_sample = np.random.choice(sample, size=n, replace=True)
    bootstrap_skewness[i] = skewness(bootstrap_sample)

# Probability that skewness < 1
prob_skew_less_1 = np.mean(bootstrap_skewness < 1)

# Bootstrap density estimate for skewness
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_skewness, bins=30, density=True, alpha=0.7, 
         edgecolor='black')
plt.axvline(x=1, color='r', linestyle='--', label='Threshold (skewness = 1)')
plt.xlabel('Skewness coefficient')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Skewness Coefficient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nd) Bootstrap for skewness coefficient:")
print(f"Bootstrap mean skewness: {np.mean(bootstrap_skewness):.4f}")
print(f"Bootstrap std skewness: {np.std(bootstrap_skewness):.4f}")
print(f"P(skewness < 1): {prob_skew_less_1:.4f}")
print(f"95% CI: [{np.percentile(bootstrap_skewness, 2.5):.4f}, "
      f"{np.percentile(bootstrap_skewness, 97.5):.4f}]")

# e) Bootstrap for sample median distribution
B_median = 1000
bootstrap_medians = np.zeros(B_median)
for i in range(B_median):
    bootstrap_sample = np.random.choice(sample, size=n, replace=True)
    bootstrap_medians[i] = np.median(bootstrap_sample)

# True distribution of median for exponential(1)
# For exponential, the median of sample of size n follows a known distribution
# Using CLT approximation for median
median_true = np.log(2)  # Population median for exponential(1)
median_se = 1 / (2 * np.exp(-median_true) * np.sqrt(n))  # Asymptotic SE

# Compare bootstrap with kernel density estimate
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_medians, bins=30, density=True, alpha=0.7, 
         edgecolor='black', label='Bootstrap estimate')

# Add KDE for bootstrap medians
kde = stats.gaussian_kde(bootstrap_medians)
x_kde = np.linspace(min(bootstrap_medians), max(bootstrap_medians), 1000)
plt.plot(x_kde, kde(x_kde), 'g-', linewidth=2, label='Bootstrap KDE')

# CLT approximation for median
x_norm = np.linspace(min(bootstrap_medians), max(bootstrap_medians), 1000)
clt_median_pdf = stats.norm.pdf(x_norm, median_true, median_se)
plt.plot(x_norm, clt_median_pdf, 'r--', linewidth=2, 
         label='Asymptotic normal (median)')

plt.xlabel('Sample median')
plt.ylabel('Density')
plt.title('Distribution of Sample Median: Bootstrap vs Asymptotic')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\ne) Sample median distribution:")
print(f"True population median (exponential(1)): {median_true:.4f}")
print(f"Sample median from data: {sample_median:.4f}")
print(f"Bootstrap mean median: {np.mean(bootstrap_medians):.4f}")
print(f"Bootstrap std median: {np.std(bootstrap_medians):.4f}")
print(f"Asymptotic std median: {median_se:.4f}")
print(f"95% CI for median (bootstrap): [{np.percentile(bootstrap_medians, 2.5):.4f}, "
      f"{np.percentile(bootstrap_medians, 97.5):.4f}]")

# Additional summary statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Original sample: min={np.min(sample):.3f}, max={np.max(sample):.3f}")
print(f"Sample mean: {sample_mean:.3f}")
print(f"Sample variance: {np.var(sample, ddof=1):.3f}")
print(f"Sample skewness: {sample_skewness:.3f}")
print(f"True skewness for exponential(1): 2")
print(f"P(skewness < 1): {prob_skew_less_1:.3f}")
