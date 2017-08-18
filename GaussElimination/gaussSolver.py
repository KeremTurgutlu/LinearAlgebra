import numpy as np
# A and b are numpy arrays
def gaussSolver(A, b):
    solution_idx = np.arange(len(A))
    n = A.shape[0]
    M = np.append(A, b, axis = 1)

    # Eliminate
    for k in range(n):
        order = sorted(np.argsort(M[k:][:, k]), reverse=True)
        solution_idx[k:] = solution_idx[k:][order]
        M[k:] = M[k:][order]

        for j in range(k+1, n):
            if M[j][k] != 0:
                multiplier = M[j][k]/float(M[k][k])
                M[j] = -(multiplier)*M[k] + M[j]

    #Solve with backwards elimination
    x = []
    for r in reversed(range(n)):
        x.append(M[r][-1]/M[r][r])
        for k in reversed(range(r)):
            M[k] = -(M[k][r] / M[r][r])*M[r] + M[k]

    x = np.array(x)

    return x[np.argsort(solution_idx)]


A = np.array([[5, 2, 1],
             [2, 8, 1],
     [5, 2, 1]])

b = np.array([[2],[12],[10]])


gaussSolver(A, b)
