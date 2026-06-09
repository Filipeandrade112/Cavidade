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

def apply_dirichlet_eigen(K, M, vertices, a, b):
    """Aplica Dirichlet (u=0) nas bordas da elipse."""
    boundary_nodes = []
    
    # Equação da elipse: (x/a)^2 + (y/b)^2 = 1
    for i, (x, y) in enumerate(vertices):
        r_val = (x/a)**2 + (y/b)**2
        # Tolerância ligeiramente maior para a aproximação poligonal da malha
        if np.isclose(r_val, 1.0, atol=1e-2): 
            boundary_nodes.append(i)

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
    """Calcula os autovalores analíticos (kc^2) resolvendo funções de Mathieu."""
    import scipy.special as sp
    import scipy.optimize as opt
    
    f = np.sqrt(a**2 - b**2)
    xi_0 = np.arccosh(a / f)
    
    def equacao_raizes(kc, m, tipo_modo):
        q = (kc * f / 2)**2
        ce_val, ce_der = sp.mathieu_modcem1(m, q, xi_0)
        return ce_val if tipo_modo == 'TM' else ce_der

    def encontra_kc(m, n, tipo_modo):
        r_eq = (a + b) / 2.0
        kc_guess = sp.jn_zeros(m, n)[n-1] / r_eq if tipo_modo == 'TM' else sp.jnp_zeros(m, n)[n-1] / r_eq
        kcs = np.linspace(kc_guess * 0.5, kc_guess * 2.0, 500)
        valores = [equacao_raizes(k, m, tipo_modo) for k in kcs]
        mudancas_sinal = np.where(np.diff(np.sign(valores)))[0]
        
        if len(mudancas_sinal) < n:
            return np.nan # Se falhar, retorna NaN para não quebrar a tabela inteira
            
        idx_raiz = mudancas_sinal[n-1]
        return opt.brentq(equacao_raizes, kcs[idx_raiz], kcs[idx_raiz+1], args=(m, tipo_modo))

    analytical_TE = []
    for (m, n) in te_modes:
        kc = encontra_kc(m, n, 'TE')
        analytical_TE.append((kc**2, m, n))
        
    analytical_TM = []
    for (m, n) in tm_modes:
        kc = encontra_kc(m, n, 'TM')
        analytical_TM.append((kc**2, m, n))

    te_sorted = sorted(analytical_TE, key=lambda x: x[0])
    tm_sorted = sorted(analytical_TM, key=lambda x: x[0])

    return te_sorted, tm_sorted

def relative_error(numerical, analytical):
    return abs(numerical - analytical) / analytical * 100