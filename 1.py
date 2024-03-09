import numpy as np
from scipy.stats import poisson, geom, binom, expon, pareto, uniform, norm

# Завдання 1
poisson_values = poisson.rvs(mu=5, size=200)
geom_values = geom.rvs(p=0.3, size=200)
binom_values = binom.rvs(n=10, p=0.4, size=200)

# Завдання 2
expon_values = expon.rvs(scale=2, size=200)
pareto_values = pareto.rvs(b=3, size=200)
uniform_values = uniform.rvs(loc=3, scale=4, size=200)
norm_values = norm.rvs(loc=10, scale=2, size=200)

# Завдання 3
num_contracts = 10
capital = 1000000  # Початковий капітал

# Розрахунок ймовірності банкрутства
def bankruptcy_probability(capital, num_contracts):
    return 1 - norm.cdf(capital, loc=num_contracts * 100000, scale=np.sqrt(num_contracts * 100000))

probability_bankruptcy = bankruptcy_probability(capital, num_contracts)

# Розрахунок капіталу, щоб ймовірність банкрутства була менше 5%
target_probability = 0.05
required_capital = norm.ppf(1 - target_probability, loc=num_contracts * 100000, scale=np.sqrt(num_contracts * 100000))

# Завдання 4
mean_fire = 9
mean_loss = 5000
reserve_fund = 105000

# Розрахунок ймовірності банкрутства
lambda_fire = mean_fire / 12
probability_bankruptcy_fire = 1 - poisson.cdf(reserve_fund / mean_loss, lambda_fire)

# Завдання 5
S0 = 100
mu = 0.02
sigma = 0.1
Wt = np.random.normal(0, 1, 20)
St_values = S0 * np.exp((mu - (sigma ** 2) / 2) * np.arange(1, 21) + sigma * np.sqrt(1) * Wt)

# Виведення результатів
print("Poisson Values:", poisson_values)
print("Geometric Values:", geom_values)
print("Binomial Values:", binom_values)
print("Exponential Values:", expon_values)
print("Pareto Values:", pareto_values)
print("Uniform Values:", uniform_values)
print("Normal Values:", norm_values)

print("\nProbability of Bankruptcy:", probability_bankruptcy)
print("Required Capital for Probability < 5%:", required_capital)

print("\nProbability of Bankruptcy (Fire Insurance):", probability_bankruptcy_fire)

print("\nStock Prices:", St_values)
