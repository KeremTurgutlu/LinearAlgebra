# A and b are numpy arrays
def gaussSolver(A, b):
    import numpy as np
    n = A.shape[0]
    M = np.append(A, b, axis = 1)
    # Eliminate
    for k in range(n):
        M[k:] = M[k:][sorted(np.argsort(M[k:][:, k]), reverse= True)]
        for j in range(k+1, n):
            if M[j][k] != 0:
                M[j] = -(M[j][k]/M[k][k])*M[k] + M[j]
    #Solve with backwards elimination
    x = []
    for r in reversed(range(n)):
        x.append(M[r][-1]/M[r][r])
        for k in reversed(range(r)):
            M[k] = -(M[k][r] / M[r][r])*M[r] + M[k]
    return(x)