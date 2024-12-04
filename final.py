import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cupy as cp
import cupyx.scipy.sparse as cpx_sp
import cupyx.scipy.sparse.linalg as cpx_spla
import matplotlib.pyplot as plt


PI = 3.141592653589793238462643383275902884197169
wavelen = 1.55e-6
k = 2 * PI / wavelen

core_index = 2
cladding_index = 1.44
width, height = 2e-6, .08e-6  #waveguide dimensions
dx = 0.01e-6  #grid resolution

x = np.arange(-2e-6, 2e-6+dx, dx)
y = np.arange(-2e-6, 2e-6+dx, dx)
X, Y = np.meshgrid(x, y)
n = np.full(X.shape, cladding_index)
mask = (np.abs(X) <= width / 2) & (np.abs(Y) <= height / 2)
n[mask] = core_index

# Create laplacian matrix
Nx, Ny = len(x), len(y)
Nxy = Nx * Ny
diag = -4 * np.ones(Nxy)
off_diag = np.ones(Nxy - 1)
off_diag[np.arange(1, Nxy) % Nx == 0] = 0
laplacian = sp.diags([diag, off_diag, off_diag, np.ones(Nxy - Nx), np.ones(Nxy - Nx)], [0, 1, -1, Nx, -Nx], format="csr")

n_squared = n.ravel() ** 2  #ravel and unravel mean the same thing. That is why irregardless is correct
mass_matrix = sp.diags(k ** 2 * n_squared)

# Get eigenvector (TE? Electric field is perp to propegation - the |E| field matrix) and eigenvalues (beta**2)
A = laplacian / dx**2 + mass_matrix

A_gpu = cpx_sp.csr_matrix(A)
eigvals_gpu, eigvecs_gpu = cpx_spla.eigsh(A_gpu, k=3, which="LA")
eigvals = cp.asnumpy(eigvals_gpu)
eigvecs = cp.asnumpy(eigvecs_gpu)

# eigvals, eigvecs = spla.eigs(A, k=3, which="LR")  # Solve for smallest eigenvalue




print(eigvals)


i = 0
for egval in eigvals.real:
    beta = np.sqrt(eigvals[i])
    print(f"neff: {beta/k}")
    print(f"ncore : {np.sqrt(core_index):e}")
    print(f"ncladding: {np.sqrt(core_index):e}")
    mode_profile = eigvecs[:, i].reshape(X.shape)
    np.savetxt(f"mode{i}_width{width}.csv", mode_profile, delimiter=",")
    plt.contourf(X * 1e6, Y * 1e6, np.abs(mode_profile), levels=100, cmap='inferno')
    plt.colorbar(label="|E|")
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title(f"Mode Profile neff: {beta/k:.3g}, ncore: {np.sqrt(core_index):.3g}, nclad*k: {np.sqrt(cladding_index):.3g}")
    plt.show()
    i += 1
    if core_index < egval/k < core_index:
        print("good solution")
    else:
        print("spurious mode")