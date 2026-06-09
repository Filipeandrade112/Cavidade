import numpy as np
from meshpy.triangle import MeshInfo, build

def generate_mesh(order, h, R):
    """Gera uma malha triangular para um guia de onda circular de raio R."""
    mesh_info = MeshInfo()

    # Define a quantidade de pontos no contorno para que os segmentos tenham aproximadamente o tamanho 'h'
    perimetro = 2 * np.pi * R
    num_pontos = max(20, int(perimetro / h))  # Pelo menos 20 pontos para não ficar um polígono feio

    # Gera as coordenadas (x, y) dos pontos na borda do círculo
    theta = np.linspace(0, 2 * np.pi, num_pontos, endpoint=False)
    pontos = [(R * np.cos(t), R * np.sin(t)) for t in theta]

    # Cria as conexões (linhas) entre os pontos adjacentes para fechar o círculo
    facetas = [(i, (i + 1) % num_pontos) for i in range(num_pontos)]

    mesh_info.set_points(pontos)
    mesh_info.set_facets(facetas)

    # Ajuste do volume (área) máximo para controlar o tamanho dos elementos internos
    max_volume = (np.sqrt(3)/4) * h**2
    mesh = build(mesh_info, max_volume=max_volume)

    vertices = np.array(mesh.points)
    elementos = np.array(mesh.elements)

    return vertices, elementos