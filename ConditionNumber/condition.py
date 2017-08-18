import numpy as np

A = np.arange(1, 101).reshape(10, 10)
b = np.ones((10, 1))

noise = np.random.rand(10, 10)
noise = (0.001 - (-0.001))*noise - 0.00111

A_new = A+noise

# Solve for new b
x = np.linalg.inv(A_new).dot(b)

# Check the condition number for A

Q = A.T.dot(A)
print(Q)
eigvals, eigvec = np.linalg.eigh(Q, "U")
r = np.linalg.matrix_rank(Q)
sing_vals = np.sqrt(eigvals[-r:])

condition_number = max(sing_vals) / min(sing_vals)

print(condition_number)