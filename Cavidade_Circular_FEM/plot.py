import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

def plot_modes(vertices, elementos, eigenvectors, mode_type, modes_plotted, num_modes=5, save_path=None):
    """
    Plota a malha e os primeiros num_modes modos com um mapa de cores (contourf).
    mode_type: "TE" ou "TM"
    modes_plotted: lista de tuplas (m,n) dos modos correspondentes ao eigenvector
    num_modes: quantos modos plotar (após a malha)
    """

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()

    # Primeiro subplot: malha
    triangles = elementos[:, :3]
    axs[0].triplot(vertices[:, 0], vertices[:, 1], triangles, color='black')
    axs[0].plot(vertices[:, 0], vertices[:, 1], 'o', color='blue', markersize=2)
    axs[0].set_title(f'Malha (P1) - {mode_type}')
    axs[0].set_xlabel('$x$ (m)')
    axs[0].set_ylabel('$y$ (m)')
    axs[0].set_aspect('equal')

    # Gerar grid para interpolação
    x_min, x_max = vertices[:,0].min(), vertices[:,0].max()
    y_min, y_max = vertices[:,1].min(), vertices[:,1].max()
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    XI, YI = np.meshgrid(xi, yi)

    # Níveis de cor fixos de 0 a 1
    niveis_cor = np.linspace(0, 1, 21)

    # Plotar os modos com contourf (preenchimento)
    for i in range(num_modes):
        eigenvector = eigenvectors[:, i]
        m, n = modes_plotted[i]

        # Interpolar
        ZI = griddata(vertices, eigenvector, (XI, YI), method='cubic')

        # Aplicar o módulo (valores positivos) e normalizar entre 0 e 1 por gráfico
        ZI_abs = np.abs(ZI)
        ZI_norm = ZI_abs / np.nanmax(ZI_abs) # nanmax evita erros nas bordas da interpolação

        cs = axs[i+1].contourf(XI, YI, ZI_norm, levels=niveis_cor, cmap='jet', vmin=0, vmax=1)
        axs[i+1].set_title(f"{mode_type}_{m}{n}")
        axs[i+1].set_xlabel('$x$ (m)')
        axs[i+1].set_ylabel('$y$ (m)')
        axs[i+1].set_aspect('equal')

        # Adicionar colorbar
        fig.colorbar(cs, ax=axs[i+1])

    plt.tight_layout()

    # --- NOVO: Salvar a imagem se um caminho for passado ---
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico {mode_type} salvo com sucesso em: {save_path}")
        
    plt.show()
