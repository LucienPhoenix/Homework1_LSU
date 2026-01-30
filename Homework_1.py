# Homework - Python, NumPy, Git/GitHub, Linear Algebra, Plotting

# **Student:** Lucien
# **Date:** January 2025 / 2026

## 1. Git & GitHub setup (done outside notebook)

# - Created repository on GitHub (e.g. `python-homework-2025` or similar)
# - Cloned it locally using VS Code
# - Created branch: `feature/homework-1`
# - Working in this branch → all code below


# ==============================================================
### 2. BASIC LIST OPERATIONS


list_a = [3, 4, 6, 10, 87, 12, 54]

# (a) max, min, sum
print("Maximum:", max(list_a))
print("Minimum:", min(list_a))
print("Sum:", sum(list_a))

# (b) sorted version
sorted_a = sorted(list_a)
print("Sorted ascending:", sorted_a)

# (c) print each element with for loop
print("Elements one by one:")
for num in list_a:
    print(num)


# =============================================================

### 3. PRIME NUMBERS

def is_prime(n):
    """Return True if n is prime, False otherwise"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# (b) primes below 100
primes_below_100 = [n for n in range(2, 100) if is_prime(n)]
print("Primes below 100:", primes_below_100)
print("There are", len(primes_below_100), "primes.")


# (c) function to find all primes below n
def find_primes_below(n):
    return [x for x in range(2, n) if is_prime(x)]


# (d) primes below 1000
primes_1000 = find_primes_below(1000)
print("\nNumber of primes below 1000:", len(primes_1000))
print("First 10:", primes_1000[:10])
print("Last 10 :", primes_1000[-10:])

# ==============================================================

### 4.  3-D NUMPY ARRAY

import numpy as np


def create_3d_array(n, data='zeros'):
    """
    Create (n × 3 × 3) numpy array
    data can be: 'zeros', 'ones', 'random'
    """
    if data == 'zeros':
        return np.zeros((n, 3, 3), dtype=int)
    elif data == 'ones':
        return np.ones((n, 3, 3), dtype=int)
    elif data == 'random':
        return np.random.randint(1, 101, size=(n, 3, 3))
    else:
        raise ValueError("data must be 'zeros', 'ones' or 'random'")


# (b) n=10, zeros
arr10 = create_3d_array(10, 'zeros')
print("Shape of arr10:", arr10.shape)
print("First slice:\n", arr10[0])

# (c) n=5, random 1–100
arr5_rand = create_3d_array(5, 'random')
print("\nShape of arr5_rand:", arr5_rand.shape)
print("First slice:\n", arr5_rand[0])

# ===============================================================

###  5. SOLVING A SYSTEM OF LINEAR EQUATION with Numpy

#   3a − 5b − 5c = 10
#        5b + c  =  8
#    a + 3b + 6c = -5

A = np.array([
    [3, -5, -5],
    [0, 5, 1],
    [1, 3, 6]
])

b = np.array([10, 8, -5])

# Solve Ax = b
x = np.linalg.solve(A, b)

print("Solution:")
print("a =", x[0])
print("b =", x[1])
print("c =", x[2])

# Verification
print("\nVerification:")
print("Eq1:", 3 * x[0] - 5 * x[1] - 5 * x[2], "≈ 10")
print("Eq2:", 5 * x[1] + x[2], "≈ 8")
print("Eq3:", x[0] + 3 * x[1] + 6 * x[2], "≈ -5")

# ===============================================================

### 6.

import matplotlib.pyplot as plt
import numpy as np

# (a) Plot
x = np.linspace(-5, 5, 400)
y = 0.1 * x ** 3 - x ** 2 + 5

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='y = 0.1x³ − x² + 5', color='royalblue', linewidth=2.5)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', lw=0.7)
plt.axvline(0, color='black', lw=0.7)

# (b) Find max & min in interval [-5,5]
idx_max = np.argmax(y)
idx_min = np.argmin(y)

x_max, y_max = x[idx_max], y[idx_max]
x_min, y_min = x[idx_min], y[idx_min]

plt.plot(x_max, y_max, 'ro', ms=10, label=f'max ≈ ({x_max:.2f}, {y_max:.2f})')
plt.plot(x_min, y_min, 'go', ms=10, label=f'min ≈ ({x_min:.2f}, {y_min:.2f})')

plt.annotate(f'max', xy=(x_max, y_max), xytext=(x_max - 1, y_max + 2),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate(f'min', xy=(x_min, y_min), xytext=(x_min + 0.5, y_min - 3),
             arrowprops=dict(facecolor='green', shrink=0.05))

plt.title("Function y = 0.1x³ − x² + 5    |   x ∈ [-5, 5]")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()