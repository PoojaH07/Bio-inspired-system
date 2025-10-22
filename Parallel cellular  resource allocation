import numpy as np
from collections import Counter

def get_neighbors(grid, i, j, neighborhood_type="4"):
    H, W = grid.shape
    neighbors = []

    # 4-neighborhood: top, bottom, left, right
    if neighborhood_type == "4":
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                neighbors.append(grid[ni, nj])
    
    # 8-neighborhood: all surrounding cells
    elif neighborhood_type == "8":
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    neighbors.append(grid[ni, nj])
    
    return neighbors

def image_processing(I, max_iter=10, neighborhood_type="4", update_rule="average", tol=0.01):
    I = np.array(I, dtype=float)  # ensure float for averaging
    H, W = I.shape
    Grid = I.copy()
    t = 0

    while t < max_iter:
        NewGrid = Grid.copy()
        changed = 0

        for i in range(H):
            for j in range(W):
                Neigh = get_neighbors(Grid, i, j, neighborhood_type)
                if update_rule == "average":  # Denoising
                    NewGrid[i,j] = np.mean(Neigh + [Grid[i,j]])
                elif update_rule == "majority":  # Segmentation
                    NewGrid[i,j] = Counter(Neigh + [Grid[i,j]]).most_common(1)[0][0]

                if abs(NewGrid[i,j] - Grid[i,j]) > tol:
                    changed += 1

        Grid = NewGrid
        t += 1

        if changed / (H*W) < tol:
            break

    return Grid

# ------------------ Example Usage ------------------

# Input image as a 2D pixel array (0-255 grayscale)
I = np.array([
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [10, 10, 10, 200, 200],
    [50, 50, 50, 150, 150],
    [50, 50, 50, 150, 150]
], dtype=float)

# Apply denoising (average)
I_final = image_processing(I, max_iter=20, neighborhood_type="4", update_rule="average", tol=1)

print("Original Image:\n", I)
print("Processed Image:\n", I_final)
