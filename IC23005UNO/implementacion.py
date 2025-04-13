# 2025 - Universidad de El Salvador - Ingenieria en Desarrollo de Software
# Cálculo Numérico para el Desarrollo de Aplicaciones
# Examen Corto 1 - Grupo Teórico 2
# Implementación de métodos numéricos para sistemas de ecuaciones lineales y no lineales

import numpy as np
import IC23005UNO as ic # Importar la librería. Se requite la intalaci+on con: pip install
import time

# --- Función auxiliar para imprimir resultados de forma ordenada ---
def print_solution(method_name, x, A=None, b=None, expected_x=None, time_taken=None):
    """Imprime la solución y métricas relevantes."""
    print(f"\n--- {method_name} ---")
    if x is not None:
        print("Solución x:")
        # Imprimir con formato consistente (6 decimales, sin notación científica por defecto)
        with np.printoptions(precision=6, suppress=True):
             print(x)
        # Calcular y mostrar residuo si A y b están disponibles
        if A is not None and b is not None:
            # Asegurarse de que b sea columna para el cálculo del residuo
            b_col = b.reshape(-1, 1) if b.ndim == 1 else b
            residuo = A @ x - b_col
            norma_residuo = np.linalg.norm(residuo)
            print(f"Norma del residuo ||Ax - b||: {norma_residuo:.3e}") # Notación científica para valores pequeños
        # Calcular y mostrar error si la solución esperada está disponible
        if expected_x is not None:
             # Asegurarse de que expected_x sea columna
             expected_x_col = expected_x.reshape(-1, 1) if expected_x.ndim == 1 else expected_x
             error = x - expected_x_col
             norma_error = np.linalg.norm(error)
             print(f"Norma del error ||x - x_esperado||: {norma_error:.3e}")
    else:
        print("El método no retornó una solución (None). Verificar advertencias/errores en la consola.")

    # Imprimir tiempo de ejecución si se proporciona
    if time_taken is not None:
         print(f"(Tiempo de ejecución: {time_taken:.4f}s)")


# --- Datos del problema lineal ---
A_lin = np.array([[ 4., -1.,  1.],
                  [-1.,  4.25, 2.75],
                  [ 1.,  2.75, 3.5]])
# b puede ser 1D o 2D (columna), los métodos lo manejan, pero la verificación necesita columna
b_lin = np.array([ 4., 6., 7.25])
b_lin_col = b_lin.reshape(-1, 1)
x_esperado_lin = np.array([[1.0], [1.0], [1.0]]) # Solución conocida

# --- Datos para métodos iterativos (matriz diagonalmente dominante para asegurar convergencia) ---
A_iter = np.array([[10., -1.,  2.],
                   [ 1., 11., -1.],
                   [ 2.,  1., 10.]])
b_iter = np.array([ 6., 25., -11.]) # b puede ser 1D aquí
b_iter_col = b_iter.reshape(-1,1)
x_esperado_iter = np.array([[1.0], [2.0], [-1.0]]) # Solución conocida

# --- Función para búsqueda de raíces ---
def func_raiz(x):
    # Ejemplo 1: Polinomio
    # return x**3 - x - 2.0 # Raíz aprox 1.5213797
    # Ejemplo 2: Trigonométrica/Exponencial
    return np.cos(x) - x      # Raíz aprox 0.7390851

print(f"--- Ejecutando ejemplos para IC23005UNO v{ic.__version__} ---")
current_date = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"--- Fecha y hora actual: {current_date} ---")

print("\n=====================================================")
print("SISTEMAS DE ECUACIONES LINEALES DIRECTOS")
print("=====================================================\n")
print("Matriz A:\n", A_lin)
print("Vector b:\n", b_lin_col) # Mostrar b como columna
print("Solución Esperada x:\n", x_esperado_lin)

start = time.perf_counter() # Usar perf_counter para mejor precisión
x_gauss = ic.eliminacion_gauss(A_lin, b_lin_col)
end = time.perf_counter()
print_solution("Método de Eliminación de Gauss", x_gauss, A_lin, b_lin_col, x_esperado_lin, time_taken=end-start)

start = time.perf_counter()
x_gj = ic.eliminacion_gauss_jordan(A_lin, b_lin_col)
end = time.perf_counter()
print_solution("Método de Eliminación de Gauss-Jordan", x_gj, A_lin, b_lin_col, x_esperado_lin, time_taken=end-start)

start = time.perf_counter()
# Cramer necesita b 1D internamente, pero podemos pasar la columna igual
x_cramer = ic.regla_cramer(A_lin, b_lin_col)
end = time.perf_counter()
# Advertencia: Cramer es lento, el tiempo puede ser mayor
print_solution("Regla de Cramer", x_cramer, A_lin, b_lin_col, x_esperado_lin, time_taken=end-start)

start = time.perf_counter()
P, L, U = ic.descomposicion_lu(A_lin)
time_decomp = time.perf_counter() - start
# x_lu = None
# time_solve = 0
if P is not None:
    print("\n--- Descomposición LU ---")
    print("P:\n", P)
    print("L:\n", L)
    print("U:\n", U)
    print(f"Verificación ||PA - LU||: {np.linalg.norm(P @ A_lin - L @ U):.2e}")
    # start_solve = time.perf_counter()
    # x_lu = ic.solve_lu(P, L, U, b_lin_col)
    # time_solve = time.perf_counter() - start_solve
else:
     print("\n--- Descomposición LU Falló ---")
print_solution("Solución usando LU", A_lin, b_lin_col, x_esperado_lin, time_taken=time_decomp)
#print_solution("Solución usando LU", x_lu, A_lin, b_lin_col, x_esperado_lin, time_taken=time_decomp + time_solve)

print("\n=====================================================")
print("SISTEMAS DE ECUACIONES LINEALES ITERATIVOS")
print("=====================================================\n")
print("Matriz A (Diagonalmente Dominante):\n", A_iter)
print("Vector b:\n", b_iter_col)
print("Solución Esperada x:\n", x_esperado_iter)

tol_iter = 1e-8
max_iter_num = 100
print(f"(Tolerancia={tol_iter}, Max_Iter={max_iter_num})")

start = time.perf_counter()
# Pasar b como vector 1D o columna, el método lo maneja
x_jac = ic.metodo_jacobi(A_iter, b_iter, tol=tol_iter, max_iter=max_iter_num)
end = time.perf_counter()
print_solution("Jacobi", x_jac, A_iter, b_iter_col, x_esperado_iter, time_taken=end-start)

start = time.perf_counter()
x_gs = ic.metodo_gauss_seidel(A_iter, b_iter, tol=tol_iter, max_iter=max_iter_num)
end = time.perf_counter()
print_solution("Gauss-Seidel", x_gs, A_iter, b_iter_col, x_esperado_iter, time_taken=end-start)

print("\n=====================================================")
print("SISTEMAS DE ECUACIONES NO LINEALES")
print("=====================================================\n")
print("Función: f(x) = cos(x) - x")
a_raiz = 0.0
b_raiz = 1.0
tol_raiz = 1e-7
print(f"Intervalo: [{a_raiz}, {b_raiz}], Tolerancia: {tol_raiz}")

start = time.perf_counter()
raiz_b = ic.metodo_biseccion(func_raiz, a_raiz, b_raiz, tol=tol_raiz)
end = time.perf_counter()
print("\n--- Bisección ---")
if raiz_b is not None:
    print(f"Raíz encontrada: {raiz_b:.7f}")
    print(f"Valor f(raíz): {func_raiz(raiz_b):.2e}")
else:
    print("El método no encontró la raíz.")
print(f"(Tiempo de ejecución: {end - start:.4f}s)")

print("\n=====================================================")
print("ACÁ TERMINAN LOS EJEMPLOS DE IMPLEMENTACIÓN")
print("=====================================================\n")