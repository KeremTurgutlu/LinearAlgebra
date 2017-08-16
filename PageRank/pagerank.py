import numpy as np

A = np.array(
    [
        [0,  1 , 0, 0], [1,  0, 0, 0], [0,  0, 0, 1], [0,  0, 1, 0]
    ]
)

v = np.array([0.25, 0.25, 0.25, 0.25]).reshape(4, 1)



print(A)
print(v)

print(A.dot(v))
print(A.dot(A.dot(v)))
print(A.dot(A.dot(A.dot(v))))
print(A.dot(A.dot(A.dot(A.dot(v)))))


print(np.linalg.matrix_power(A, 100).dot(v))
print(np.linalg.matrix_power(A, 200).dot(v))