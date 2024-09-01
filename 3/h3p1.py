import numpy as np

'''(a):'''
def LU_decomposition(matrix):
    n = len(matrix)
    # 初始化 L 和 U 矩阵
    L = np.eye(n)
    U = np.zeros((n, n))

    for j in range(n):
        # 计算 U 的第 j 行
        for i in range(j, n):
            U[j, i] = matrix[j, i] - np.dot(L[j, :j], U[:j, i])

        # 计算 L 的第 j+1 列
        for i in range(j+1, n):
            L[i, j] = (matrix[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

    return L, U

# 测试
matrix = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]])
L, U = LU_decomposition(matrix)
print("L矩阵：")
print(L)
print("\nU矩阵：")
print(U)

# 验证 LU 分解结果
M = np.dot(L, U)
print("\nL * U 矩阵：")
print(M)

'''(b)'''

def solve_double_backward_substitution(L, U, b):
    n = len(b)
    x = np.zeros(n)
    y = np.zeros(n)

    # 解 Ly = b，先解下三角方程
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # 解 Ux = y，再解上三角方程
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# 给定的L，U矩阵和b向量
b = np.array([-4, 3, 9, 7])

# 解线性方程组
x = solve_double_backward_substitution(L, U, b)
print("解得的x向量：")
print(x)

'''(c)'''
def partial_pivoting_LU_decomposition(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = np.copy(matrix).astype(float)  # 将矩阵转换为浮点数类型

    for k in range(n):
        max_row = np.argmax(np.abs(U[k:, k])) + k

        # 交换行
        U[[k, max_row]] = U[[max_row, k]]

        # 记录置换的行
        temp_row = np.copy(L[k, :k])
        L[k, :k] = L[max_row, :k]
        L[max_row, :k] = temp_row

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U



# 测试
matrix = np.array([[0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]])
L, U = partial_pivoting_LU_decomposition(matrix)
print("L矩阵：")
print(L)
print("\nU矩阵：")
print(U)

# 验证 LU 分解结果
M = np.dot(L, U)
print("\nL * U 矩阵：")
print(M)

'''(continued)'''

def partial_pivoting_LU_decomposition(matrix):
    n = len(matrix)
    L = np.eye(n)
    U = np.copy(matrix).astype(float)  # 将矩阵转换为浮点数类型
    swaps = []  # 交换列表，用于记录行交换顺序

    for k in range(n):
        max_row = np.argmax(np.abs(U[k:, k])) + k

        # 交换行
        U[[k, max_row]] = U[[max_row, k]]
        L[[k, max_row]] = L[[max_row, k]]  # 交换L矩阵的行
        swaps.append((k, max_row))  # 记录行交换顺序

        for i in range(k + 1, n):
            if U[k, k] != 0:  # 判断除数是否为零
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]
            else:
                raise ValueError("Encountered zero pivot, unable to continue LU decomposition.")

    return L, U, swaps

def gaussian_elimination_with_row_pivoting(L, U, b, swaps):
    n = len(b)
    x = np.zeros(n)

    # 根据交换列表进行行交换
    for swap in swaps:
        b[swap[0]], b[swap[1]] = b[swap[1]], b[swap[0]]

    # 正向替换 (Forward substitution)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # 反向替换 (Backward substitution)
    for i in range(n - 1, -1, -1):
        if U[i, i] != 0:  # 判断除数是否为零
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        else:
            raise ValueError("Encountered zero pivot, unable to continue Gaussian elimination.")

    return x

# 测试
matrix = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]])
b = np.array([-4, 3, 9, 7])

# 部分枢轴LU分解
L, U, swaps = partial_pivoting_LU_decomposition(matrix)
print("L矩阵：")
print(L)
print("\nU矩阵：")
print(U)
print("\n交换列表：")
print(swaps)

# 求解方程组
x = gaussian_elimination_with_row_pivoting(L, U, b, swaps)
print("\n利用带旋转的LU分解解得的x向量：")
print(x)

# 定义系数矩阵A和常数向量b
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]])
b = np.array([-4, 3, 9, 7])

# 使用numpy.linalg.solve求解方程组
x = np.linalg.solve(A, b)

# 打印解向量x
print("利用solve函数解得的x向量：")
print(x)