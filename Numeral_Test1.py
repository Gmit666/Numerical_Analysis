import numpy as np

def get_square_matrix():
    """Prompts the user for input and creates a square matrix."""
    while True:
        try:
            rows = float(input("Enter the number of rows and columns (square matrix): "))
            if rows <= 0 or rows % 1 != 0:
                raise ValueError("Matrix size must be a positive integer.")
            matrix = np.array(
                [[float(x) for x in input(f"Enter row {i+1} elements (separated by spaces): ").split()]
                 for i in range(int(rows))])
            if matrix.shape == (int(rows), int(rows)):
                return matrix
            else:
                raise ValueError("Input is not a square matrix.")
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def matrix_vector_multiply(matrix, vector):
    """Multiply a matrix by a vector."""
    return np.dot(matrix, vector)

def divide_matrices(matrix1, matrix2):
    """Divides each element of matrix1 by the corresponding element of matrix2."""
    try:
        result = np.divide(matrix1, matrix2)
        return result
    except ZeroDivisionError:
        print("Error: Division by zero encountered.")
        return None
def calculate_determinant(matrix):
    """Calculates the determinant of a square matrix with specified precision."""
    det = np.round(np.linalg.det(matrix), 3)  # Calculate and round to 3 decimal places
    return det
def max_norm(matrix):
    """Calculates the max norm of a matrix."""
    row_sums = np.sum(np.abs(matrix), axis=1)
    max_norm_value = np.max(row_sums)
    return max_norm_value

def matrix_type(matrix):
    """Calculates the max norm of a matrix and prints the result."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    max_norm_value = max_norm(matrix)

    print(f"Max norm of the matrix: {round(max_norm_value, 5)}")



def inverse_matrix(matrix):
    """Calculates the inverse matrix using elementary row operations and prints iterations."""

    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Check if matrix is invertible (non-zero determinant)
    det = np.round(np.linalg.det(matrix), 5)  # Calculate and round to 5 decimal places
    if det == 0:
        print("The matrix is not invertible (determinant is zero).")
        return None

    # Create identity matrix of the same size
    n = len(matrix)
    identity = np.eye(n)

    # Perform elimination steps
    for i in range(n):
        # Find pivot element and check for zero
        pivot_row = i
        pivot = abs(matrix[i, i])
        for j in range(i + 1, n):
            if abs(matrix[j, i]) > pivot:
                pivot_row = j
                pivot = abs(matrix[j, i])
        if pivot == 0:
            raise ValueError("Matrix is not invertible.")

        # Swap rows if necessary
        if pivot_row != i:
            matrix[[i, pivot_row]] = matrix[[pivot_row, i]]
            identity[[i, pivot_row]] = identity[[pivot_row, i]]
            print(f"\nSwapped rows {i + 1} and {pivot_row + 1}:")
            print(matrix)
            print("Elementary matrix:")
            print(identity)

        # Normalize the pivot row
        pivot_element = matrix[i, i]
        matrix[i, :] /= pivot_element
        identity[i, :] /= pivot_element
        print(f"\nNormalized row {i + 1}:")
        print(matrix)
        print("Elementary matrix:")
        print(identity)

        # Eliminate elements above and below the pivot
        for j in range(n):
            if j != i:
                factor = matrix[j, i]
                matrix[j, :] -= factor * matrix[i, :]
                identity[j, :] -= factor * identity[i, :]
                print(f"\nSubtracted {factor} times row {i + 1} from row {j + 1}:")
                print(matrix)
                print("Elementary matrix:")
                print(identity)

    print("\nInverse matrix:")
    print(identity)
    return identity
def gaussian_elimination(matrix, b):
    """Solves a system of linear equations using Gaussian elimination with guided pivot selection."""
    n = len(matrix)
    augmented_matrix = np.hstack((matrix, b.reshape(n, 1)))  # Reshape RHS vector

    for i in range(n):
        # Suggest pivot and allow user override
        max_row = i
        for j in range(i + 1, n):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[max_row, i]):
                max_row = j
        suggestion = f"Suggested pivot: row {max_row+1}, column {i+1}"
        choice = input(f"\nPivot selection: {suggestion} (y/n) or choose row,column (e.g., 2,3): ")
        if choice.lower() != 'y':
            try:
                row, col = map(int, choice.split(','))
                if not (0 <= row < n and 0 <= col < n):
                    raise ValueError
                max_row = row - 1  # adjust for user input starting from 1
            except ValueError:
                print("Invalid input. Using suggested pivot.")
        # Swap rows
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        for j in range(i + 1, n):
            multiplier = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, :] -= multiplier * augmented_matrix[i, :]

    # Back substitution
    solution = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solution[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, :i], solution[:i])) / augmented_matrix[i, i]
    return solution

def main():
    """Provides a menu for user choices and calls appropriate functions."""
    while True:
        print("\nMatrix Calculator")
        print("1. Get Square Matrix")
        print("2. Calculate Determinant")
        print("3. Calculate Inverse Matrix")
        print("4. Solve System of Equations (Gaussian Elimination)")
        print("5. Matrix Max Norm")
        print("6. Matrix Multiplication")
        print("7. Matrix Division")
        print("0. Exit")
        choice = input("Enter your choice: ")

        try:
            if choice == '0':
                break

            elif choice == '1':
                matrix = get_square_matrix()
                print("Square matrix:")
                print(matrix)

            elif choice == '2':
                matrix = get_square_matrix()
                det = calculate_determinant(matrix)
                print("Determinant:", det)

            elif choice == '3':
                matrix = get_square_matrix()
                if np.linalg.det(matrix) == 0:
                    print("Error: Matrix is not invertible. Determinant is 0.")
                else:
                    inv = inverse_matrix(matrix)

            elif choice == '4':
                matrix = get_square_matrix()
                b = np.array([float(x) for x in input("Enter RHS vector elements (separated by spaces): ").split()])
                solution = gaussian_elimination(matrix, b)
                print("Solution:", solution)

            elif choice == '5':
                matrix = get_square_matrix()
                matrix_type(matrix)

            elif choice == '6':
                matrix1 = get_square_matrix()
                matrix2 = get_square_matrix()
                matrix3 = matrix_vector_multiply(matrix1, matrix2)
                print("Multiplication Result:")
                print(matrix3)

            elif choice == '7':
                matrix1 = get_square_matrix()
                matrix2 = get_square_matrix()
                matrix3 = divide_matrices(matrix1, matrix2)
                print("Division Result:")
                print(matrix3)

            else:
                print("Invalid choice. Please choose from the available options.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

