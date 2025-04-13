# 2025 - Universidad de El Salvador - Ingenieria en Desarrollo de Software
# Cálculo Numérico para el Desarrollo de Aplicaciones
# Examen Corto 1 - Grupo Teórico 2
# Implementación de métodos numéricos para sistemas de ecuaciones no lineales
"""
Solucionadores para sistema de ecuaciones no lineales.
Esta versión incluye únicamente búsqueda de raíces por biseccion (f(x)=0).
"""
import numpy as np

def metodo_biseccion(func, a, b, tol=1e-7, max_iter=100):
    """
    Encuentra una raíz de f(x)=0 en [a, b] usando el método de bisección.
    Se requiere f(a) * f(b) < 0 y su continuidad.
    """
    try:
        if not callable(func): raise TypeError("'func' debe ser callable.")
        if a >= b: raise ValueError("'a' debe ser menor que 'b'.")

        fa = func(a)
        fb = func(b)
        if np.sign(fa) == np.sign(fb):
            print(f"Error: f(a) y f(b) tienen el mismo signo en [{a}, {b}]. f(a)={fa:.2e}, f(b)={fb:.2e}")
            return None
        if np.isclose(fa, 0.0): return a
        if np.isclose(fb, 0.0): return b

        n_iter = 0
        c = a # Inicializar c por si el bucle no se ejecuta
        while n_iter < max_iter:
            c = a + (b - a) / 2.0
            fc = func(c)

            if (b - a) / 2.0 < tol or np.isclose(fc, 0.0, atol=tol):
                return c

            if np.sign(fc) == np.sign(fa):
                a = c
                fa = fc
            else:
                b = c
            n_iter += 1

        print(f"No hay convergencia después de {max_iter} iteraciones.")
        return c # Devolver la última aproximación
    except Exception as e:
        print(f"Error en metodo_biseccion: {e}")
        return None