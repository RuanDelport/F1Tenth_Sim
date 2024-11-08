import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the range for x
x = np.linspace(-10, 10, 1000)

# Normal distribution parameters
mu = 0
sigma = 1
normal_dist = norm.pdf(x, mu, sigma)

# Non-normal distribution with skew, multimodality, and heavy tails
# Example: a mixture of two normal distributions (bimodal) with one peak skewed
skewed_peak_1 = norm.pdf(x, -3, 1) * 0.5
skewed_peak_2 = norm.pdf(x, 2, 0.5) * 0.8
heavy_tail_dist = skewed_peak_1 + skewed_peak_2

# Combined distribution
combined_dist = normal_dist + heavy_tail_dist

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, normal_dist, label="Normal Distribution", linestyle="--")
plt.plot(x, heavy_tail_dist, label="Non-Normal Distribution")
plt.plot(x, combined_dist, label="Combined Distribution", color="red", linewidth=2)
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Combination of Normal and Non-Normal Distributions")
plt.legend()
plt.grid(True)
plt.show()
    