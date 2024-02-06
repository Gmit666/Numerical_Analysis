from random import randint
def create_square_matrix(num):
    if(num < 2):
        raise ValueError("Number is not sufficiant for square matrix")

    else:
        matrix = [[0 for x in range(num)] for y in range(num)]
        return matrix

def fill_random_matrix(matrix):
    num = len(matrix)
    for i in range(num):
        for j in range(num):
            matrix[i][j] = randint(0,100)
    return matrix

def fill_matrix(matrix):
    num = len(matrix)
    print("Enter values for matrix:")
    for i in range(num):
        for j in range(num):
            matrix[i][j] = int(input())
    return matrix

def display_all_matrix(a,b,c,d):
    print("Matrix A:", a)
    print("Matrix B:", b)
    print("Matrix C - Addition result:", c)
    print("Matrix D - Multiplication result:", d)

def matrix_addition(matrix1, matrix2):
    if(len(matrix1) != len(matrix2)):
        raise ValueError("Matrix must have the same length.")
    num = len(matrix1)
    c = create_square_matrix(num)
    for i in range(num):
        for j in range(num):
            c[i][j] = matrix1[i][j] + matrix2[i][j]
    return c

def matrix_multiplication(matrix1, matrix2):
    if (len(matrix1) != len(matrix2)):
        raise ValueError("Matrix must have the same length.")
    num = len(matrix1)
    c = create_square_matrix(num)
    result = 0
    for i in range(num):
        for j in range(num):
            for k in range(num):
                c[i][j] += matrix1[i][k] * matrix2[k][j]
    return c

def main():
    print("Fill matrix manually or with random values (0 / 1):")
    choice = int(input())
    while(choice < 0 or choice > 1):
        choice = int(input("Enter valid input (0 / 1):"))
    print("Enter size of square matrix:")
    size = int(input())

    match choice:
        case 0:
            a = fill_matrix(create_square_matrix(size))
            b = fill_matrix(create_square_matrix(size))
        case 1:
            a = fill_random_matrix(create_square_matrix(size))
            b = fill_random_matrix(create_square_matrix(size))

    c = matrix_addition(a, b)
    d = matrix_multiplication(a, b)
    display_all_matrix(a,b,c,d)

main()
