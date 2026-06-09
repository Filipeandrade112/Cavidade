import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.special as sp_special  # Importante para as Funções de Bessel
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

def apply_dirichlet_eigen(K, M, vertices, R):
    """Aplica condições de contorno de Dirichlet (u = 0 nas bordas do círculo)."""
    boundary_nodes = []
    
    # Encontra os nós que estão na borda circular (r = R)
    for i, (x, y) in enumerate(vertices):
        r = np.sqrt(x**2 + y**2)
        # Usamos isclose com uma tolerância para evitar erros de arredondamento da malha
        if np.isclose(r, R, atol=1e-5):
            boundary_nodes.append(i)

    # Zera as linhas e colunas e coloca 1 na diagonal principal
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

def analytical_eigenvalues(R, te_modes, tm_modes):
    """Calcula os autovalores analíticos (kc^2) para um guia circular."""
    analytical_TE = []
    for (n, m) in te_modes:
        # Raiz da derivada da função de Bessel para TE
        raiz = sp_special.jnp_zeros(n, m)[m-1]
        kc2 = (raiz / R)**2
        analytical_TE.append((kc2, n, m))
        
    analytical_TM = []
    for (n, m) in tm_modes:
        # Raiz da função de Bessel pura para TM
        raiz = sp_special.jn_zeros(n, m)[m-1]
        kc2 = (raiz / R)**2
        analytical_TM.append((kc2, n, m))

    # Organiza do menor para o maior autovalor (frequência)
    te_eigenvalues_sorted = sorted(analytical_TE, key=lambda x: x[0])
    tm_eigenvalues_sorted = sorted(analytical_TM, key=lambda x: x[0])

    return te_eigenvalues_sorted, tm_eigenvalues_sorted

def relative_error(numerical, analytical):
    return abs(numerical - analytical) / analytical * 100