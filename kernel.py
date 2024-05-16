import numpy as np

# Identity Kernel
KerI = np.array([[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]])

# Blur Kernel
KerB = 1 / 9 * np.array([[1, 1, 1],
                         [1, 1, 1],
                         [1, 1, 1]])

# Sobel Horizontal Kernel
KerSH = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])

# Sobel Vertical Kernel
KerSV = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])

# Prewitt Horizontal Kernel
KerPH = np.array([[-1, 0, 1],
                  [-1, 0, 1],
                  [-1, 0, 1]])

# Prewitt Vertical Kernel
KerPV = np.array([[-1, -1, -1],
                  [0, 0, 0],
                  [1, 1, 1]])

# Gaussian Blur 7x7 Kernel
KerGB7 = np.array([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 2, 1, 1]
])
KerGB7 = KerGB7 / np.sum(KerGB7)

# Gaussian Blur 5x5 Kernel
KerGB = 1 / 256 * np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]])
