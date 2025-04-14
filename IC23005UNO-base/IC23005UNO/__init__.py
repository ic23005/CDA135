"""
IC23005UNO: Librería para resolver ecuaciones lineales y no lineales con diferentes metodos.
"""

# Define la versión de la librería
__version__ = "1.0.1" # Incrementa la versión por el cambio en la llamada de los métodos: metodo_gauss_jordan y regla_crammer 

# Importa las funciones desde los módulos
from .sistemas_lineales import (
    eliminacion_gauss,
    eliminacion_gauss_jordan,
    regla_crammer,
    descomposicion_lu,
    metodo_jacobi,
    metodo_gauss_seidel,
)
from .sistemas_no_lineales import (
    metodo_biseccion,
)

# Define qué se importa con 'from IC23005UNO import *'
__all__ = [
    # Sistemas Lineales
    'eliminacion_gauss',
    'eliminacion_gauss_jordan',
    'regla_cramer',
    'descomposicion_lu',
    'metodo_jacobi',
    'metodo_gauss_seidel',
    # Sistemas NO Lineales
    'metodo_biseccion',
    # Version
    '__version__',
]