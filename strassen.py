import numpy as np
import random
import math
import time
import sys

##################################################
# MATRIX GENERATION Functions & Associated Helpers
##################################################

# Returns two n * n matrice where sizes = power of 2
def matGen(n):
    A = []
    B = []

    for i in range(n):
        for j in range(n):
            num1 = random.randint(0, 2)
            num2 = random.randint(0, 2)
            # create new row
            if j == 0:
                A.append([num1])
                B.append([num2])
            # add to row
            else:
                A[i].append(num1)
                B[i].append(num2)


    if checkPow2(n) == False:
        return makePow2Mats(A, B)
    else:
        return A, B

# Returns matrices padded with 0s where sizes = power of 2
def makePow2Mats(A, B):
    size = 0
    n = len(A)

    for i in range(n, 2*n):
        for j in range(i):
            A[j].append(0)
            B[j].append(0)
        A.append([0] * (i+1))
        B.append([0] * (i+1))
        
        if checkPow2(i+1) == True:
            size = i + 1
            break

    return A, B

# Checks if num is a power of two
def checkPow2(num):
    return float(math.floor(math.log(num, 2))) == float(math.log(num, 2))

#################################################################
# MULTIPLCATION FUNCTIONS (Naive & Strassen) & Associated Helpers
#################################################################

# Creates n * n matrix of 0s for product output (not time consuming)
def makeEmptyListMat(n):
    mat = [0] * n
    for i in range(n):
        mat[i] = [0] * n
    return mat

# NAIVE Multiplication for two n * n matrices
def naive(A, B):
    # multiply two nxn matrices using the naive method
    n = len(A)
    C = makeEmptyListMat(n)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C

# STRASSEN Multiplication for two n * n matrices
def startStrassen(size, cross):
    A, B = matGen(size)
    return strassen(A, B, cross)

def strassen(A, B, cross):
    LENGTH = len(A)

    if (LENGTH <= cross): # base case/crossover point
        return naive(A, B)

    DIV = LENGTH // 2

    a, e = getSubMat(A, 0, DIV, 0), getSubMat(B, 0, DIV, 0) # top lefts
    b, f = getSubMat(A, 0, DIV, DIV), getSubMat(B, 0, DIV, DIV) # top rights
    c, g = getSubMat(A, DIV, LENGTH, 0), getSubMat(B, DIV, LENGTH, 0) # bottom lefts
    d, h = getSubMat(A, DIV, LENGTH, DIV), getSubMat(B, DIV, LENGTH, DIV) # bottom rights

    p1 = strassen(a, addSub(f, h, "-"), cross)
    p2 = strassen(addSub(a, b, "+"), h, cross)
    p3 = strassen(addSub(c, d, "+"), e, cross)
    p4 = strassen(d, addSub(g, e, "-"), cross)
    p5 = strassen(addSub(a, d, "+"), addSub(e, h, "+"), cross)
    p6 = strassen(addSub(b, d, "-"), addSub(g, h, "+"), cross)
    p7 = strassen(addSub(a, c, "-"), addSub(e, f, "+"), cross)

    topleft = addSub(addSub(addSub(p5, p4, "+"), p2, "-"), p6, "+")
    # topleft = addSub(p5, addSub(p4, addSub(p2, p6, "+"), "-"), "+")
    topright = addSub(p1, p2, "+")
    bottomleft = addSub(p3, p4, "+")
    bottomright = addSub(addSub(addSub(p1, p5, "+"), p3, "-"), p7, "-")
    # bottomright = addSub(p1, addSub(p5, addSub(p3, p7, "-"),"-"),"+")

    C = mergeMats(topleft, topright, bottomleft, bottomright)
    
    return C

def mergeMats(A, B, C, D):
    # merge together matrices top left, top right, bottom left, bottom right
    # topleft = A, topright = B, bottomleft = C, bottomRight = D
    n = 2 * len(A)
    outArr = makeEmptyListMat(n)

    for i in range(n):
        for j in range(n):
            if i < (n // 2) and j < (n // 2):
                outArr[i][j] = A[i][j]
            elif i < (n // 2) and j >= (n // 2):
                outArr[i][j] = B[i][j - (n // 2)]
            elif i >= (n // 2) and j < (n // 2):
                outArr[i][j] = C[i - (n // 2)][j]
            else:
                outArr[i][j] = D[i - (n // 2)][j - (n // 2)]

    return outArr

# Extract equal dimensional submatrix from argument matrix
# from "start" index up to but not including "end" index
def getSubMat(A, rowStart, rowEnd, colStart):
    n = rowEnd - rowStart
    C = makeEmptyListMat(n)

    for i in range(n):
        for j in range(n):
            C[i][j] = A[rowStart + i][colStart + j]
    
    return C

# Add or Subtract Matices
def addSub(A, B, op):
    n = len(A)
    C = makeEmptyListMat(n)

    if op == "+":
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] + B[i][j]

    elif op == "-":
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] - B[i][j]
    
    return C

#######################################
# TESTING & Associated Helper Functions
#######################################
def printNice(A):
    n = len(A)
    print("    ", end="")
    for i in range(n):
        if i < 10:
            print(str(i), end="  ")
        else:
            print(str(i), end=" ")
    print()
    for i in range(n):
        if i < 10:
            print(" ", end="")
        print(str(i) + ": ", end="")
        for j in range(n):
            if A[i][j] < 10:
                print(str(A[i][j]), end="  ")
            else:
                print(str(A[i][j]), end=" ")
        print()

# TESTING STRASSENS ALGO:
    # 2:25 for sz = 1000 x 1000, cross-over @ n = 64
    # 2:30 for sz = 1000 x 1000, cross-over @ n = 128

def tests():
    print()
    # test for these cross-over points
    crosses = [32, 64, 128, 256, 512, 1024]
    # test for these sizes
    sizes = [512, 1024, 1536, 2048, 4096, 6144, 8192]
    for size in sizes:
        for cross in crosses:
            if cross < size:
                start = time.time()
                A, B = matGen(size)
                C = strassen(A, B, cross)
                print(str(size) + " x " + str(size) + " Matrices, " + str(cross) + ": Cross-Over Pt")
                print("Runtime: " + str(time.time() - start) + " seconds")
                print()
            else:
                continue


def flip(p):
    # simulate a weighted coin flip with probability p
    return 1 if random.random() <= p else 0

def randomGraph(prob, n=1024):
    A = []

    for i in range(n):
        A.append([])
        for j in range(n):
            num = flip(prob) # the number to put in A[i][j]
            if i == j:
                num = 0
            if j < i:
                # print("len(A[j])"+str(len(A[j])))
                # print("j: "+str(j))
                # print("i: "+str(i))
                num = A[j][i]
            A[i].append(num)

    return A

def getNumTriangles(A):
    # sums the diagonal of A, where A is the original random graph cubed
    # divides sum by 6, returns number of triangles
    n = len(A)
    triangles = 0
    for i in range(n):
        triangles += A[i][i]
    triangles = triangles / 6
    return triangles


# g = [[0, 1, 0, 1, 1], 
#     [1, 0, 1, 1, 1],
#     [0, 1, 0, 1, 0], 
#     [1, 1, 1, 0, 0], 
#     [1, 1, 0, 0, 0]]

# gSquared = strassen(g, g, 64)
# gCubed = strassen(g, gSquared, 64)
# printNice(gSquared)
# printNice(gCubed)
# print(getNumTriangles(gCubed))

probs = [0.01, 0.02, 0.03, 0.04, 0.05]
CROSS = 64
for p in probs:
    graph = randomGraph(p, 128)
    # printNice(graph)
    graphNumPy = np.array(graph)
    graph2 = strassen(graph, graph, CROSS)
    graph2NumPy = np.dot(graphNumPy,graphNumPy)
    if np.array_equal(np.array(graph2), graph2NumPy):
        print("Graph squared correctly")
    else:
        print("Graph squared incorrectly")
    graph3 = strassen(graph, graph2, CROSS)
    graph3NumPy = np.dot(graphNumPy,graph2NumPy)
    if np.array_equal(np.array(graph3), graph3NumPy):
        print("Graph cubed correctly")
    else:
        print("Graph cubed incorrectly")
    # print("Cube of random graph:")
    # printNice(graph3)
    print("Number of triangles: " + str(getNumTriangles(graph3)))


# tests()

# SIZE = 1000
# CROSS = 64
# A, B = matGen(SIZE)
# C = strassen(A, B, CROSS)
# A1, B1 = np.array(A), np.array(B)

# print()
# if np.array_equal(np.array(C), np.dot(A1, B1)):
#     print("Good Product")
# else:
#     print("Bad Product")

# print("Size: " + str(len(A)) + " x " + str(len(A)))
# print()

# For testing Strassen Multiplication
# print("A:")
# printNice(A)
# print()
# print("B:")
# printNice(B)
# print()
# print("Strassen Multiplication:")
# printNice(C)
# print()
# print("NumPy Multiplication")
# printNice(np.dot(A1, B1))

#######################################

# TESTING FOR:
    # Correct random matrix generation where size is power of 2
    # Correct NAIVE matrix multiplication

# 2:15 running time for sz = 1000 with numPy matrix creations and product comparison
# 2:00 running time for sz = 1000 w/o numPy arrays (matrices generated instantly)

# SIZE = 65
# A, B = matGen(SIZE)
# A1, B1 = np.array(A), np.array(B)

# if np.array(naive(A, B)).all() == np.dot(A1, B1).all():
#     print("Good Product")
# else:
#     print("Bad Product")

# print("Size: " + str(len(A)) + " x " + str(len(A)))

#######################################

# TESTING FOR:
    # Sub-Matrices correctly generated

# SIZE = 6
# A, B, a, b, c, d, e, f, g, h = startStrassen(SIZE, CROSS)

# printNice(A)
# print()
# print("top left")
# printNice(a)
# print()
# print("top right")
# printNice(b)
# print()
# print("bottom left")
# printNice(c)
# print()
# print("bottom right")
# printNice(d)
# print()
# printNice(B)
# print()
# print("top left")
# printNice(e)
# print()
# print("top right")
# printNice(f)
# print()
# print("bottom left")
# printNice(g)
# print()
# print("bottom right")
# printNice(h)

if __name__ == "__main__":
    # take input from command line
    # ./strassen 0 dimension inputfile

    #sys.argv[0] # this is 0
    dims = sys.argv[1]
    filename = sys.argv[2]

    A = []
    B = []
    with open(filename, "r") as f:
        i = 0
        j = 0
        usingB = False
        A.append([])
        for line in f:
            if j < dims:
                if not usingB:
                    A[i].append(int(line))
                else:
                    B[i].append(int(line))
                j += 1
            else:
                j = 0
                i += 1
                if not usingB:
                    A.append([])
                else:
                    B.append([])
            if i >= dims:
                i = 0
                j = 0
                usingB = True
    
    C = strassen(A, B, CROSS)
    
    # output the diagonal of C
    for i in range(dims):
        print(C[i][i])
        
    print()
