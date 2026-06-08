import numpy as np
from meshpy.triangle import MeshInfo, build

def generate_mesh(order, h, a, b):
    """Gera uma malha triangular para um guia de onda retangular de tamanho a x b."""
    mesh_info = MeshInfo()

    pontos = [
        (0.0, 0.0),
        (a, 0.0),
        (a, b),
        (0.0, b)
    ]

    facetas = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0)
    ]

    mesh_info.set_points(pontos)
    mesh_info.set_facets(facetas)

    # Ajuste do volume máximo para controlar o tamanho dos elementos
    max_volume = (np.sqrt(3)/4) * h**2
    mesh = build(mesh_info, max_volume=max_volume)

    vertices = np.array(mesh.points)
    elementos = np.array(mesh.elements)

    return vertices, elementos
