import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Definir los parámetros del problema
E = 2.1e11   # Módulo de elasticidad del material (Pa)
F = 10000   # Fuerza vertical aplicada en la parte superior de la barra (N)
num_iterations = 100   # Número de iteraciones del descenso de gradiente

# Definir la función que genera la malla de elementos finitos a lo largo de la barra
def generate_mesh(points):
    tri = Delaunay(points)
    return tri.simplices

# Definir la función que calcula la deformación de la barra para una forma dada
def compute_deformation(mesh):
    # Calcular el área y la longitud de cada elemento finito
    areas = []
    lengths = []
    for i in range(mesh.shape[0]):
        p1, p2, p3 = mesh[i]
        a = np.linalg.norm(points[p1] - points[p2]) / 2
        b = np.linalg.norm(points[p2] - points[p3]) / 2
        c = np.linalg.norm(points[p3] - points[p1]) / 2
        s = (a + b + c) / 2
        areas.append(np.sqrt(s * (s - a) * (s - b) * (s - c)))
        lengths.append(a + b + c)
    # Definir la matriz de rigidez global
    K_global = np.zeros((len(points), len(points)))
    for i in range(mesh.shape[0]):
        p1, p2, p3 = mesh[i]
        x1, y1, z1 = points[p1]
        x2, y2, z2 = points[p2]
        x3, y3, z3 = points[p3]
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = z2 - z3
        c2 = z3 - z1
        c3 = z1 - z2
        d1 = x3 - x2
        d2 = x1 - x3
        d3 = x2 - x1
        V = np.array([[d1, d2, d3], [b1, b2, b3], [c1, c2, c3]])
        k = V
