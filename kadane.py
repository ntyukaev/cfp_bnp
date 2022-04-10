import sys
from utils import read


# Function to find maximum continuous
# maximum sum in the array
def kadane(v):
    # Stores current and maximum sum
    currSum = 0

    maxSum = -sys.maxsize - 1

    # Traverse the array v
    for i in range(len(v)):

        # Add the value of the
        # current element
        currSum += v[i]

        # Update the maximum sum
        if (currSum > maxSum):
            maxSum = currSum
        if (currSum < 0):
            currSum = 0

    # Return the maximum sum
    return maxSum


# Function to find the maximum
# submatrix sum
def maxSubmatrixSum(A):
    # Store the rows and columns
    # of the matrix
    r = len(A)
    c = len(A[0])

    # Create an auxiliary matrix
    # Traverse the matrix, prefix
    # and initialize it will all 0s
    prefix = [[0 for i in range(c)]
              for j in range(r)]

    # Calculate prefix sum of all
    # rows of matrix A[][] and
    # store in matrix prefix[]
    for i in range(r):
        for j in range(c):

            # Update the prefix[][]
            if (j == 0):
                prefix[i][j] = A[i][j]
            else:
                prefix[i][j] = A[i][j] + prefix[i][j - 1]

    # Store the maximum submatrix sum
    maxSum = -sys.maxsize - 1

    #  Iterate for starting column
    for i in range(c):

        # Iterate for last column
        for j in range(i, c):

            # To store current array
            # elements
            v = []

            # Traverse every row
            for k in range(r):

                # Store the sum of the
                # kth row
                el = 0

                # Update the prefix
                # sum
                if (i == 0):
                    el = prefix[k][j]
                else:
                    el = prefix[k][j] - prefix[k][i - 1]

                # Push it in a vector
                v.append(el)

            # Update the maximum
            # overall sum
            maxSum = max(maxSum, kadane(v))

    # Print the answer
    print(maxSum)


# Driver Code
# matrix = [[0, -2, -7, 0],
#           [9, 2, -6, 2],
#           [-4, 1, -4, 1],
#           [-1, 8, 0, -2]]

matrix = read('examples/test.txt')

# Function Call
maxSubmatrixSum(matrix)
