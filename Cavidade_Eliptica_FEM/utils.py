import numpy as np

def shape_functions_P1(xi, eta):
    """Funções de forma para elementos P1."""
    N = [
        1 - xi - eta,
        xi,
        eta
    ]
    dN = [
        np.array([-1, -1]),
        np.array([1, 0]),
        np.array([0, 1])
    ]
    return N, dN

def verificar_dados(vertices, elementos, ordem):
    """Verifica a validade dos dados da malha."""
    if not isinstance(vertices, (np.ndarray, list)):
        raise ValueError("vertices deve ser um array numpy ou uma lista.")
    if not isinstance(elementos, (np.ndarray, list)):
        raise ValueError("elementos deve ser um array numpy ou uma lista.")

    vertices = np.array(vertices)
    elementos = np.array(elementos)

    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices deve ser uma matriz 2D com duas colunas (x e y).")

    num_nos = len(vertices)
    esperado_nos_por_elemento = 3  # P1 tem 3 nós por elemento

    for elemento in elementos:
        if len(elemento) != esperado_nos_por_elemento:
            raise ValueError(f"Para P{ordem}, cada elemento deve ter exatamente {esperado_nos_por_elemento} nós.")
        if any(idx < 0 or idx >= num_nos for idx in elemento):
            raise ValueError("Índices dos elementos fora do intervalo válido.")

    print(f"Vertices e elementos estão configurados corretamente para ordem P{ordem}.")
