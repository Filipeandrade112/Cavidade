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

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
pasta_graficos = os.path.join(diretorio_atual, "Graficos")
os.makedirs(pasta_graficos, exist_ok=True)

def run_eigen_simulation_eliptica_P1(a=0.05, b=0.03, h=0.001):
    # Modos Pares (Even) para guias elípticos (m=ordem angular, n=raiz radial)
    te_modes = [
        (0, 1),  # eTE_01
        (1, 1),  # eTE_11 (Modo fundamental dependendo da excentricidade)
        (2, 1),  # eTE_21
        (0, 2),  # eTE_02
        (1, 2)   # eTE_12
    ]

    tm_modes = [
        (0, 1),  # eTM_01
        (1, 1),  # eTM_11
        (2, 1),  # eTM_21
        (0, 2),  # eTM_02
        (1, 2)   # eTM_12
    ]

    # Geração da malha adaptada para a elipse
    vertices, elementos = generate_mesh(order=1, h=h, a=a, b=b)
    print(f"Malha gerada: {len(vertices)} vértices, {len(elementos)} elementos.")

    verificar_dados(vertices, elementos, ordem=1)

    K_TE, M_TE = assemble_eigen_system_TE(vertices, elementos, ordem=1)
    print("Matrizes K_TE e M_TE montadas para modos TE.")

    K_TM, M_TM = assemble_eigen_system_TM(vertices, elementos, ordem=1)
    print("Matrizes K_TM e M_TM montadas para modos TM.")

    # Aplicar Dirichlet no contorno elíptico somente para TM
    K_TM_bc, M_TM_bc = apply_dirichlet_eigen(K_TM, M_TM, vertices, a=a, b=b)
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
        print(f"Convergência não alcançada para TE.")

    try:
        eigen_TM, eigenvectors_TM = solve_eigen_system(K_TM_bc, M_TM_bc, num_eigen_TM, tol=1e-4, maxiter=100000)
        print(f"{len(eigen_TM)} autovalores TM computados.")
    except spla.ArpackNoConvergence as e:
        eigen_TM = e.eigenvalues
        eigenvectors_TM = e.eigenvectors
        print(f"Convergência não alcançada para TM.")

    def filter_eigenvalues(vals, vecs):
        mask = (vals > 1) & (~np.isclose(vals, 1.0, atol=1e-12))
        return vals[mask], vecs[:, mask]

    eigen_TE_filtered, eigenvectors_TE_filtered = filter_eigenvalues(eigen_TE, eigenvectors_TE)
    eigen_TM_filtered, eigenvectors_TM_filtered = filter_eigenvalues(eigen_TM, eigenvectors_TM)

    # Calcular autovalores usando Funções de Mathieu
    analytical_TE, analytical_TM = analytical_eigenvalues(a, b, te_modes, tm_modes)

    def match_modes(anal_modes, eigen_vals, eigen_vecs):
        matched_vals = []
        matched_vecs = []
        for (anal_val, m, n) in anal_modes:
            if np.isnan(anal_val):
                continue # Pula se o solver de Mathieu falhou
            idx = np.argmin(np.abs(eigen_vals - anal_val))
            matched_vals.append((anal_val, eigen_vals[idx], m, n, idx))
        
        reordered_vecs = np.zeros((eigen_vecs.shape[0], len(matched_vals)))
        for i, mv in enumerate(matched_vals):
            reordered_vecs[:, i] = eigen_vecs[:, mv[4]]

        return matched_vals, reordered_vecs

    matched_TE, eigenvectors_TE_ordered = match_modes(analytical_TE, eigen_TE_filtered, eigenvectors_TE_filtered)
    matched_TM, eigenvectors_TM_ordered = match_modes(analytical_TM, eigen_TM_filtered, eigenvectors_TM_filtered)

    print("\nTabela de Autovalores TE (Modos Pares):")
    print("Modo\tNumérico (k_c^2)\t\tAnalítico (k_c^2)\tErro (%)")
    for (anal_val, num_val, m, n, _) in matched_TE:
        erro = relative_error(num_val, anal_val)
        print(f"eTE_{m}{n}\t{num_val:.6f}\t\t{anal_val:.6f}\t\t{erro:.2f}%")

    print("\nTabela de Autovalores TM (Modos Pares):")
    print("Modo\tNumérico (k_c^2)\t\tAnalítico (k_c^2)\tErro (%)")
    for (anal_val, num_val, m, n, _) in matched_TM:
        erro = relative_error(num_val, anal_val)
        print(f"eTM_{m}{n}\t{num_val:.6f}\t\t{anal_val:.6f}\t\t{erro:.2f}%")

    # Plotar e Salvar modos TE
    te_modes_plotted = [(m, n) for (_, _, m, n, _) in matched_TE[:5]]
    caminho_te = os.path.join(pasta_graficos, "MODOS TE ELIPTICOS.png")
    plot_modes(vertices, elementos, eigenvectors_TE_ordered, "eTE", te_modes_plotted, num_modes=min(5, len(te_modes_plotted)), save_path=caminho_te)

    # Plotar e Salvar modos TM
    tm_modes_plotted = [(m, n) for (_, _, m, n, _) in matched_TM[:5]]
    caminho_tm = os.path.join(pasta_graficos, "MODOS TM ELIPTICOS.png")
    plot_modes(vertices, elementos, eigenvectors_TM_ordered, "eTM", tm_modes_plotted, num_modes=min(5, len(tm_modes_plotted)), save_path=caminho_tm)


if __name__ == "__main__":
    # a DEVE ser maior que b
    a = 0.05
    b = 0.03
    h = 0.0015 # Aumentei h um pouco para rodar mais rápido nos testes
    run_eigen_simulation_eliptica_P1(a=a, b=b, h=h)