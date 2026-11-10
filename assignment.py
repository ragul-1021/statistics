import numpy as np

A = np.array([[4, 1], [2, 3]], dtype=float)
x = np.array([1, 1], dtype=float)
tol = 1e-8
for iteration in range(1, 101):
    y = A @ x
    x_new = y / np.linalg.norm(y)
    if np.linalg.norm(x_new - x) < tol:
        x = x_new
        break
    x = x_new
eigenvalue = (x @ A @ x) / (x @ x)
print(f"Dominant eigenvalue: {eigenvalue:.4f}")
print("Eigenvector:", np.round(x, 4))