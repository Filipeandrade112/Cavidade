import numpy as np
from meshpy.triangle import MeshInfo, build

def generate_mesh(order, h, a, b):
    """Gera uma malha triangular para um guia de onda elíptico (semi-eixos a e b)."""
    mesh_info = MeshInfo()

    # Aproximação do perímetro de Ramanujan para a elipse
    perimetro = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    num_pontos = max(30, int(perimetro / h)) 

    # Coordenadas (x, y) da elipse
    theta = np.linspace(0, 2 * np.pi, num_pontos, endpoint=False)
    pontos = [(a * np.cos(t), b * np.sin(t)) for t in theta]

    facetas = [(i, (i + 1) % num_pontos) for i in range(num_pontos)]

    mesh_info.set_points(pontos)
    mesh_info.set_facets(facetas)

    max_volume = (np.sqrt(3)/4) * h**2
    mesh = build(mesh_info, max_volume=max_volume)

    vertices = np.array(mesh.points)
    elementos = np.array(mesh.elements)

    return vertices, elementos