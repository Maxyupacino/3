import numpy as np

'''(b)'''
def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n),dtype=float)
    R = np.zeros((n, n),dtype=float)

    for j in range(n):
        v = A[:, j].copy().astype(float)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

# 测试
A = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9],[4, 7, 9, 2]])
Q, R = qr_decomposition(A)

print("Q矩阵：")
print(Q)
print("\nR矩阵：")
print(R)
M = np.dot(Q,R)
print("\nA矩阵：")
print(M)

'''(c)'''
def qr_algorithm_symmetric(matrix, threshold=1e-6):
    # 首先将矩阵转换为浮点数类型
    matrix = matrix.astype(float)
    n = matrix.shape[0]

    # 初始化特征向量矩阵
    eigenvectors = np.eye(n)

    while True:
        # 进行一次QR分解
        Q, R = np.linalg.qr(matrix)

        # 计算新的矩阵
        matrix = np.dot(R, Q)

        # 更新特征向量矩阵
        eigenvectors = np.dot(eigenvectors, Q)

        # 检查是否满足停止条件
        off_diagonal = np.abs(matrix - np.diag(np.diag(matrix)))
        if np.max(off_diagonal) < threshold:
            break

    # 提取特征值
    eigenvalues = np.diag(matrix)

    return eigenvalues, eigenvectors

# 测试
matrix = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9],[4, 7, 9, 2]])
eigenvalues, eigenvectors = qr_algorithm_symmetric(matrix)

print("特征值：", eigenvalues)
print("特征向量：\n", eigenvectors)