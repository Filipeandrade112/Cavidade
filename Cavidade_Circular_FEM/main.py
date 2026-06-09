import os
import numpy as np
from mesh_generation import generate_mesh
from fem import (
    assemble_eigen_system_TE,
    assemble_eigen_system_TM,
    apply_dirichlet_eigen,
    solve_eigen_system,
    analytical_eigenvalues,
    relative_error
)
from utils import verificar_dados
from plot import plot_modes
import scipy.sparse.linalg as spla

# Pega o diretório exato onde o arquivo main.py está salvo no seu computador
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
# Junta o diretório atual com a pasta 'Graficos'
pasta_graficos = os.path.join(diretorio_atual, "Graficos")
os.makedirs(pasta_graficos, exist_ok=True)

def run_eigen_simulation_circular_P1(R=0.04, h=0.001):
    # Modos desejados na ordem clássica para guias circulares (n=azimutal, m=radial)
    te_modes = [
        (1, 1),  # TE_11 (Modo fundamental do guia circular)
        (2, 1),  # TE_21
        (0, 1),  # TE_01
        (3, 1),  # TE_31
        (1, 2)   # TE_12
    ]

    tm_modes = [
        (0, 1),  # TM_01
        (1, 1),  # TM_11
        (2, 1),  # TM_21
        (0, 2),  # TM_02
        (1, 2)   # TM_12
    ]

    # Geração da malha adaptada para o raio R
    vertices, elementos = generate_mesh(order=1, h=h, R=R)
    print(f"Malha gerada: {len(vertices)} vértices, {len(elementos)} elementos.")

    verificar_dados(vertices, elementos, ordem=1)

    # A montagem global depende apenas dos elementos e vértices da malha
    K_TE, M_TE = assemble_eigen_system_TE(vertices, elementos, ordem=1)
    print("Matrizes K_TE e M_TE montadas para modos TE.")

    K_TM, M_TM = assemble_eigen_system_TM(vertices, elementos, ordem=1)
    print("Matrizes K_TM e M_TM montadas para modos TM.")

    # Aplicar Dirichlet no contorno circular somente para TM
    K_TM_bc, M_TM_bc = apply_dirichlet_eigen(K_TM, M_TM, vertices, R=R)
    print("Condições de contorno aplicadas para modos TM.")

    K_TE_bc = K_TE.tocsc()
    M_TE_bc = M_TE.tocsc()
    K_TM_bc = K_TM_bc.tocsc()
    M_TM_bc = M_TM_bc.tocsc()

    num_eigen_TE = 20
    num_eigen_TM = 20

    try:
        eigen_TE, eigenvectors_TE = solve_eigen_system(K_TE_bc, M_TE_bc, num_eigen_TE, tol=1e-4, maxiter=100000)
        print(f"{len(eigen_TE)} autovalores TE computados.")
    except spla.ArpackNoConvergence as e:
        eigen_TE = e.eigenvalues
        eigenvectors_TE = e.eigenvectors
        print(f"Convergência não alcançada para TE: {len(eigen_TE)} autovalores convergidos.")

    try:
        eigen_TM, eigenvectors_TM = solve_eigen_system(K_TM_bc, M_TM_bc, num_eigen_TM, tol=1e-4, maxiter=100000)
        print(f"{len(eigen_TM)} autovalores TM computados.")
    except spla.ArpackNoConvergence as e:
        eigen_TM = e.eigenvalues
        eigenvectors_TM = e.eigenvectors
        print(f"Convergência não alcançada para TM: {len(eigen_TM)} autovalores convergidos.")

    # Filtragem: remover valores espúrios próximos a 1 ou menores que 1
    def filter_eigenvalues(vals, vecs):
        mask = (vals > 1) & (~np.isclose(vals, 1.0, atol=1e-12))
        return vals[mask], vecs[:, mask]

    eigen_TE_filtered, eigenvectors_TE_filtered = filter_eigenvalues(eigen_TE, eigenvectors_TE)
    eigen_TM_filtered, eigenvectors_TM_filtered = filter_eigenvalues(eigen_TM, eigenvectors_TM)

    # Calcular os autovalores analíticos usando as raízes de Bessel com base no raio R
    analytical_TE, analytical_TM = analytical_eigenvalues(R, te_modes, tm_modes)

    # Função para achar o autovalor numérico mais próximo do analítico
    def match_modes(anal_modes, eigen_vals, eigen_vecs):
        matched_vals = []
        matched_vecs = []
        for (anal_val, n, m) in anal_modes:
            # Encontrar o índice do autovalor numérico mais próximo
            idx = np.argmin(np.abs(eigen_vals - anal_val))
            matched_vals.append((anal_val, eigen_vals[idx], n, m, idx))
        
        # Extrair os vetores ordenados conforme os modos solicitados
        reordered_vecs = np.zeros((eigen_vecs.shape[0], len(matched_vals)))
        for i, mv in enumerate(matched_vals):
            idx = mv[4]
            reordered_vecs[:, i] = eigen_vecs[:, idx]

        return matched_vals, reordered_vecs

    matched_TE, eigenvectors_TE_ordered = match_modes(analytical_TE, eigen_TE_filtered, eigenvectors_TE_filtered)
    matched_TM, eigenvectors_TM_ordered = match_modes(analytical_TM, eigen_TM_filtered, eigenvectors_TM_filtered)

    # Exibir Tabela TE
    print("\nTabela de Autovalores TE (na ordem solicitada):")
    print("Modo\tNumérico (k_c^2)\t\tAnalítico (k_c^2)\tErro (%)")
    for (anal_val, num_val, n, m, _) in matched_TE:
        erro = relative_error(num_val, anal_val)
        modo = f"TE_{n}{m}"
        print(f"{modo}\t{num_val:.6f}\t\t{anal_val:.6f}\t\t{erro:.2f}%")

    # Exibir Tabela TM
    print("\nTabela de Autovalores TM (na ordem solicitada):")
    print("Modo\tNumérico (k_c^2)\t\tAnalítico (k_c^2)\tErro (%)")
    for (anal_val, num_val, n, m, _) in matched_TM:
        erro = relative_error(num_val, anal_val)
        modo = f"TM_{n}{m}"
        print(f"{modo}\t{num_val:.6f}\t\t{anal_val:.6f}\t\t{erro:.2f}%")

    # Plotar e Salvar modos TE (5 primeiros)
    te_modes_plotted = [(n, m) for (_, _, n, m, _) in matched_TE[:5]]
    caminho_te = os.path.join(pasta_graficos, "MODOS TE.png")
    plot_modes(vertices, elementos, eigenvectors_TE_ordered, "TE", te_modes_plotted, num_modes=5, save_path=caminho_te)

    # Plotar e Salvar modos TM (5 primeiros)
    tm_modes_plotted = [(n, m) for (_, _, n, m, _) in matched_TM[:5]]
    caminho_tm = os.path.join(pasta_graficos, "MODOS TM.png")
    plot_modes(vertices, elementos, eigenvectors_TM_ordered, "TM", tm_modes_plotted, num_modes=5, save_path=caminho_tm)


if __name__ == "__main__":
    # Define o Raio (R) em metros (ex: 4 cm = 0.04m) e o tamanho do elemento (h)
    R = 0.04
    h = 0.001
    run_eigen_simulation_circular_P1(R=R, h=h)