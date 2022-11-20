import numpy as np

n = 5
a = np.random.sample((n, n))
q_, r_ = np.linalg.qr(a)


# Numerical QR_decomposition
def projection_operator(A: list, B: list):
    return (np.dot(A, B) / np.dot(B, B)) * B


"""
Q matrix we can finding in GS-process - he realized it QR_decomposition
"""


def QR_decompositon(A: list):
    # Gram-Schmidt orthonormalize process
    n = len(A)
    Q = [A.T[0] / np.dot(A.T[0], A.T[0]) ** 0.5]
    for i in range(1, n):
        sum_vect = 0
        for k in range(0, i):
            sum_vect += projection_operator(A.T[i], Q[k])
        project = A.T[i] - sum_vect
        Q.append(project / np.dot(project, project) ** 0.5)
    # Q and R- matrix's
    Q = np.array(Q).T
    R = np.around(np.dot(Q.T, A), 10)
    return Q, R


u, w = QR_decompositon(a)

print("Numpy\n", q_, "\nNumeric\n", u, "\n")
print("Numpy\n", r_, "\nNumeric\n", w, "\n")
print("\nbefore QR a =\n", a, "\n", "\nAfter Numeric QR-decomposition a = QR \n", np.dot(u, w))

# For Hessenberg's Matrix we can use QR - algorithm to find eigva and eigve

"""
Use Gauss method for trial matrix
"""


# Swapping strings if diagonal elements ==0
def Swapper_matr_full(A: list):
    p = len(A)
    for i in range(p):
        if A[i][i] == 0:
            for k in range(i + 1, p):
                if A[k][i] != 0:
                    A[i], A[k] = A[k], A[i]
    return A


# Gauss Method
def Gauss_method_func_full(A: list):
    p = len(A)
    A = Swapper_matr_full(A)
    for k in range(p - 1):
        for i in range(k + 1, p):
            if A[i][k] != 0 and A[k][k] != 0:
                global pre
                pre = (A[i][k] / A[k][k])
                for j in range(k, p):
                    A[i][j] -= pre * A[k][j]
    return A


a = Gauss_method_func_full(a)


# QR - algorithm
def QR_algorithm(A: list):
    k = 10000
    multiply = 1
    for i in range(k):
        q, r = QR_decompositon(A)
        A = np.dot(r, q)
        multiply = np.dot(multiply, q)
    eigen_values = sorted([A[i][i] for i in range(len(A))])
    eigen_vect_matrix = np.array(multiply.T)
    return np.array(eigen_values), eigen_vect_matrix


pl1, pl2 = QR_algorithm(a)
eva, eve = np.linalg.eigh(a)
print("\nEigen values -- QR - algorithm \n", pl1, "\n")
print("\nEigen values -- Numpy \n", eva, "\n")
