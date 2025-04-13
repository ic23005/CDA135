# 2025 - Universidad de El Salvador - Ingenieria en Desarrollo de Software
# Cálculo Numérico para el Desarrollo de Aplicaciones
# Examen Corto 1 - Grupo Teórico 2
# Implementación de métodos numéricos para sistemas de ecuaciones lineales
"""
Solucionadores para sistemas de ecuaciones lineales Ax = b.
Incluye métodos directos e iterativos.
"""
import numpy as np

def eliminacion_gauss(A, b):
    """
    Resuelve Ax = b usando el Método de Eliminación de Gauss
    """
    try:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).reshape(-1, 1) # Asegurar que b sea columna
        n = len(b)
        if A.shape != (n, n):
            raise ValueError("La matriz A debe ser cuadrada (NxN) y coincidir con la longitud de b (N).")

        Ab = np.hstack([A, b])

        # Eliminación hacia adelante
        for i in range(n):
            pivot_row_index = i + np.argmax(np.abs(Ab[i:, i]))
            if np.isclose(Ab[pivot_row_index, i], 0.0):
                print(f"Advertencia: Matriz singular detectada en la columna {i} durante la eliminación.")
                return None
            if pivot_row_index != i:
                Ab[[i, pivot_row_index]] = Ab[[pivot_row_index, i]]

            pivot = Ab[i, i]
            for j in range(i + 1, n):
                factor = Ab[j, i] / pivot
                Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]

        # Sustitución hacia atrás
        x = np.zeros((n, 1), dtype=float)
        for i in range(n - 1, -1, -1):
            if np.isclose(Ab[i, i], 0.0):
                 print(f"Error inesperado: Pivote cero encontrado en la sustitución hacia atrás (fila {i}).")
                 return None
            sum_ax = np.dot(Ab[i, i + 1:n], x[i + 1:n, 0])
            x[i, 0] = (Ab[i, n] - sum_ax) / Ab[i, i]

        return x
    except Exception as e:
        print(f"Error en eliminacion_gauss: {e}")
        return None


def eliminacion_gauss_jorda(A, b):
    """
    Resuelve Ax = b usando el Método de Eliminación de Gauss-Jordan.
    """
    try:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).reshape(-1, 1) # Asegurar que b sea columna
        n = len(b)
        if A.shape != (n, n):
            raise ValueError("La matriz A debe ser cuadrada (NxN) y coincidir con la longitud de b (N).")

        Ab = np.hstack([A, b])

        # Eliminación
        for i in range(n):
            pivot_row_index = i + np.argmax(np.abs(Ab[i:, i]))
            if np.isclose(Ab[pivot_row_index, i], 0.0):
                print(f"Advertencia: Matriz singular detectada en la columna {i}.")
                return None
            if pivot_row_index != i:
                Ab[[i, pivot_row_index]] = Ab[[pivot_row_index, i]]

            pivot = Ab[i, i]
            Ab[i, i:] = Ab[i, i:] / pivot

            for j in range(n):
                if i != j:
                    factor = Ab[j, i]
                    Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]

        x = Ab[:, n].reshape(-1, 1)
        return x
    except Exception as e:
        print(f"Error en eliminacion_gauss_jorda: {e}")
        return None


def regla_crammer(A, b):
    """
    Resuelve Ax = b usando la Regla de Cramer. No aplica cuando N > 3
    """
    try:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).flatten()
        n = len(b)
        if A.shape != (n, n):
            raise ValueError("La matriz A debe ser cuadrada y coincidir con la longitud de b.")

        det_A = np.linalg.det(A)
        if np.isclose(det_A, 0.0):
            print("Advertencia: El determinante de A es (casi) cero. El sistema es singular.")
            return None

        x = np.zeros(n, dtype=float)
        for i in range(n):
            Ai = A.copy()
            Ai[:, i] = b
            det_Ai = np.linalg.det(Ai)
            x[i] = det_Ai / det_A

        return x.reshape(-1, 1)
    except Exception as e:
        print(f"Error en regla_crammer: {e}")
        return None


def descomposicion_lu(A):
    """
    Realiza la descomposición PA = LU usando Doolittle.
    """
    try:
        A = np.array(A, dtype=float)
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("La matriz A debe ser cuadrada.")

        L = np.eye(n, dtype=float)
        U = A.copy()
        P = np.eye(n, dtype=float)
        pivots = np.arange(n)

        for k in range(n - 1):
            pivot_index = k + np.argmax(np.abs(U[k:, k]))
            if np.isclose(U[pivot_index, k], 0.0):
                 print(f"Advertencia: Matriz singular detectada en la columna {k} durante LU.")
                 # Considerar retornar None, None, None

            if pivot_index != k:
                U[[k, pivot_index], k:] = U[[pivot_index, k], k:]
                L[[k, pivot_index], :k] = L[[pivot_index, k], :k]
                pivots[[k, pivot_index]] = pivots[[pivot_index, k]]

            pivot = U[k, k]
            if not np.isclose(pivot, 0.0):
                 for i in range(k + 1, n):
                      factor = U[i, k] / pivot
                      L[i, k] = factor
                      U[i, k:] = U[i, k:] - factor * U[k, k:]
                      U[i, k] = 0.0

        P_final = np.zeros((n, n), dtype=float)
        for i in range(n):
            P_final[i, pivots[i]] = 1.0

        if np.isclose(U[n - 1, n - 1], 0.0):
            print("Advertencia: La matriz U resultante tiene un cero en la última posición diagonal. A es singular.")

        return P_final, L, U
    except Exception as e:
        print(f"Error en descomposicion_lu: {e}")
        return None, None, None


def metodo_jacobi(A, b, x0=None, tol=1e-6, max_iter=1000, check_convergence=True):
    """
    Resuelve Ax = b usando el método iterativo de Jacobi.
    """
    try:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).flatten()
        n = len(b)
        if A.shape != (n, n):
            raise ValueError("La matriz A debe ser cuadrada y coincidir con la longitud de b.")

        diag_A = np.diag(A)
        if np.any(np.isclose(diag_A, 0.0)):
            print("Error: Jacobi requiere que todos los elementos diagonales de A sean no nulos.")
            return None

        if check_convergence:
            off_diag_sum = np.sum(np.abs(A), axis=1) - np.abs(diag_A)
            if not np.all(np.abs(diag_A) > off_diag_sum):
                print("Advertencia: Matriz A no estrictamente diagonalmente dominante. Convergencia de Jacobi no garantizada.")

        if x0 is None: x = np.zeros(n, dtype=float)
        else:
            x = np.array(x0, dtype=float).flatten()
            if len(x) != n: raise ValueError("x0 debe tener la misma longitud que b.")

        D_inv = np.diag(1.0 / diag_A)
        R = A - np.diag(diag_A)
        x_new = np.copy(x)

        for k in range(max_iter):
            x_new = D_inv @ (b - (R @ x))
            residual_norm = np.linalg.norm(x_new - x, ord=np.inf)
            if residual_norm < tol:
                return x_new.reshape(-1, 1)
            x = np.copy(x_new)

        print(f"No hay convergencia después de {max_iter} iteraciones. Norma residual: {residual_norm:.2e}")
        return None
    except Exception as e:
        print(f"Error en metodo_jacobi: {e}")
        return None


def metodo_gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=1000, check_convergence=True):
    """
    Resuelve Ax = b usando el método iterativo de Gauss-Seidel.
    """
    try:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).flatten()
        n = len(b)
        if A.shape != (n, n):
            raise ValueError("La matriz A debe ser cuadrada y coincidir con la longitud de b.")

        diag_A = np.diag(A)
        if np.any(np.isclose(diag_A, 0.0)):
            print("Error: Gauss-Seidel requiere diagonales no nulas.")
            return None

        if check_convergence:
             off_diag_sum = np.sum(np.abs(A), axis=1) - np.abs(diag_A)
             is_dd = np.all(np.abs(diag_A) > off_diag_sum)
             if not is_dd:
                  print("Advertencia: Matriz A no estrictamente diagonalmente dominante. Convergencia de Gauss-Seidel no garantizada.")

        if x0 is None: x = np.zeros(n, dtype=float)
        else:
             x = np.array(x0, dtype=float).flatten()
             if len(x) != n: raise ValueError("x0 debe tener la misma longitud que b.")

        x_old = np.copy(x)

        for k in range(max_iter):
            x_old[:] = x[:]
            for i in range(n):
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]

            residual_norm = np.linalg.norm(x - x_old, ord=np.inf)
            if residual_norm < tol:
                return x.reshape(-1, 1)

        print(f"No hay convergencia después de {max_iter} iteraciones. Norma residual: {residual_norm:.2e}")
        return None
    except Exception as e:
        print(f"Error en metodo_gauss_seidel: {e}")
        return None