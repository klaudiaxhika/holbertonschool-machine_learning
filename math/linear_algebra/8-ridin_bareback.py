#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    mat = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))] 
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            product = 0
            for k in range(len(mat2)):
                mat[i][j] += mat1[i][k] * mat2[k][j]
    return mat

mat1 = [[1, 2],
        [3, 4],
        [5, 6]]
mat2 = [[1, 2, 3, 4],
        [5, 6, 7, 8]]
print(mat_mul(mat1, mat2))

