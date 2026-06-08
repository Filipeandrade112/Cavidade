import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from utils import shape_functions_P1

def assemble_eigen_system_TE(vertices, elementos, ordem):
    """Monta as matrizes K e M para o modo TE (Hz como desconhecido)."""
    num_nos = len(vertices)
    K_TE = sp.lil_matrix((num_nos, num_nos))
    M_TE = sp.lil_matrix((num_nos, num_nos))

    # Pontos de Gauss e pesos
    pontos_gauss = [(1/3, 1/3, 0.5)]

    def shape_funcs(xi, eta):
        return shape_functions_P1(xi, eta)

    for elemento in elementos:
        indices_nos = elemento
        coordenadas_nos = vertices[indices_nos]
        n_nos = len(indices_nos)

        K_e = np.zeros((n_nos, n_nos))
        M_e = np.zeros((n_nos, n_nos))

        for (xi, eta, peso) in pontos_gauss:
            N, dN = shape_funcs(xi, eta)
            J = np.zeros((2,2))
            for i in range(n_nos):
                J[0,0] += dN[i][0]*coordenadas_nos[i][0]
                J[0,1] += dN[i][1]*coordenadas_nos[i][0]
                J[1,0] += dN[i][0]*coordenadas_nos[i][1]
                J[1,1] += dN[i][1]*coordenadas_nos[i][1]
            detJ = np.abs(np.linalg.det(J))
            invJ = np.linalg.inv(J)
            grad_N = [invJ.T @ dN[i] for i in range(n_nos)]

            for i in range(n_nos):
                for j in range(n_nos):
                    K_e[i,j] += (grad_N[i] @ grad_N[j])*detJ*peso
                    M_e[i,j] += N[i]*N[j]*detJ*peso

        for i_local, i_global in enumerate(indices_nos):
            for j_local, j_global in enumerate(indices_nos):
                K_TE[i_global, j_global] += K_e[i_local, j_local]
                M_TE[i_global, j_global] += M_e[i_local, j_local]

    return K_TE, M_TE

def assemble_eigen_system_TM(vertices, elementos, ordem):
    """Monta as matrizes K e M para o modo TM (Ez como desconhecido)."""
    num_nos = len(vertices)
    K_TM = sp.lil_matrix((num_nos, num_nos))
    M_TM = sp.lil_matrix((num_nos, num_nos))

    pontos_gauss = [(1/3, 1/3, 0.5)]

    def shape_funcs(xi, eta):
        return shape_functions_P1(xi, eta)

    for elemento in elementos:
        indices_nos = elemento
        coordenadas_nos = vertices[indices_nos]
        n_nos = len(indices_nos)

        K_e = np.zeros((n_nos, n_nos))
        M_e = np.zeros((n_nos, n_nos))

        for (xi, eta, peso) in pontos_gauss:
            N, dN = shape_funcs(xi, eta)
            J = np.zeros((2,2))
            for i in range(n_nos):
                J[0,0] += dN[i][0]*coordenadas_nos[i][0]
                J[0,1] += dN[i][1]*coordenadas_nos[i][0]
                J[1,0] += dN[i][0]*coordenadas_nos[i][1]
                J[1,1] += dN[i][1]*coordenadas_nos[i][1]
            detJ = np.abs(np.linalg.det(J))
            invJ = np.linalg.inv(J)
            grad_N = [invJ.T @ dN[i] for i in range(n_nos)]

            for i in range(n_nos):
                for j in range(n_nos):
                    K_e[i,j] += (grad_N[i] @ grad_N[j])*detJ*peso
                    M_e[i,j] += N[i]*N[j]*detJ*peso

        for i_local, i_global in enumerate(indices_nos):
            for j_local, j_global in enumerate(indices_nos):
                K_TM[i_global, j_global] += K_e[i_local, j_local]
                M_TM[i_global, j_global] += M_e[i_local, j_local]

    return K_TM, M_TM

def apply_dirichlet_eigen(K, M, vertices, a, b):
    """Aplica condições de contorno de Dirichlet (u = 0 nas bordas)."""
    boundary_nodes = [i for i, (x, y) in enumerate(vertices) 
                      if np.isclose(x, 0.0, atol=1e-6) or 
                         np.isclose(x, a, atol=1e-6) or 
                         np.isclose(y, 0.0, atol=1e-6) or 
                         np.isclose(y, b, atol=1e-6)]

    for node in boundary_nodes:
        K[node, :] = 0
        K[:, node] = 0
        M[node, :] = 0
        M[:, node] = 0
        K[node, node] = 1
        M[node, node] = 1

    return K, M

def solve_eigen_system(K, M, num_eigenvalues, tol=1e-4, maxiter=100000):
    if num_eigenvalues >= K.shape[0]:
        raise ValueError("num_eigenvalues deve ser menor que o número total de nós.")
    eigenvalues, eigenvectors = spla.eigsh(K, k=num_eigenvalues, M=M, which='SM', tol=tol, maxiter=maxiter)
    return eigenvalues, eigenvectors

def analytical_eigenvalues(a, b, te_modes, tm_modes):
    te_eigenvalues = [( (m * np.pi / a)**2 + (n * np.pi / b)**2, m, n ) for m, n in te_modes]
    tm_eigenvalues = [( (m * np.pi / a)**2 + (n * np.pi / b)**2, m, n ) for m, n in tm_modes]

    te_eigenvalues_sorted = sorted(te_eigenvalues, key=lambda x: x[0])
    tm_eigenvalues_sorted = sorted(tm_eigenvalues, key=lambda x: x[0])

    return te_eigenvalues_sorted, tm_eigenvalues_sorted

def relative_error(numerical, analytical):
    return abs(numerical - analytical) / analytical * 100
