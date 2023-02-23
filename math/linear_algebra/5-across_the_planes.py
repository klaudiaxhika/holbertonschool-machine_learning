#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    result=[]
    if [len(mat1),len(mat1[0])] != [len(mat2),len(mat2[0])]:
        return None
    for i in range(len(mat1)):
        result.append([])
        for j in range(len(mat1)):
            result[i].append(mat1[i][j]+mat2[i][j])
    return result

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6], [7, 8]]
print(add_matrices2D(mat1, mat2))
print(mat1)
print(mat2)
print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
