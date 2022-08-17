import numpy as np

def jacobi(A, b, E=0, N=25, x=None):
    """
    Resolve o Sistema de Equações Ax = b 
    pelo Método Iterativo de Jacobi
    """
    
    # Chute inicial é um vetor nulo se não for fornecido pelo usuário
    if x is None:
        x = np.zeros(len(A[0]))

    # Manipulação na Matriz dos Coeficientes A para criação da Matriz C:
    # - Elementos da Diagonal Principal são subtraídos de A
    # - Matriz é dividida pelo Elemento da Diagonal Principal correspondente de cada linha
    # - Matriz é negada
    D = np.diag(A)
    C = - (A - np.diagflat(D)) / np.vstack(D)

    # Manipulação na Matriz dos Termos Independentes b para criação da Matriz g:
    # - Matriz é dividida pelo Elemento da Diagonal Principal correspondente de cada linha
    g = b / D

    # Método de Jacobi é iterado até que seja satisfeita a precisão ou até que seja atingido o número máximo de iterações
    # Se esses dados não forem fornecidos pelo usuário, assume-se: precisão -> E = 0 e iterações max -> N = 25
    i = 0
    calc_err = 1
    while(calc_err > E and i < N):
        # Valor anterior do vetor x é armazenado em _x
        _x = x

        # Aplicação do Método de Jacobi: x = C._x + g
        x = np.dot(C, _x) + g

        # Cálculo do erro ou distância relativa: calc_err = max|x - _x| / max|x|
        calc_err = np.max(np.abs(x - _x)) / np.max(np.abs(x))

        # Incremento do número de iterações i
        i += 1

    # Função retorna o vetor x dos resultados aproximados, o erro e o número de iterações
    return (x, calc_err, i)

def convergence(A):
    """
    Aplica o Teorema do Critério das Linhas para verificar se
    há convergência no método iterativo                         
    """

    # Elementos da diagonal principal são subtraídos da Matriz de Coeficientes A
    D = np.diag(A)
    R = A - np.diagflat(D)

    # Para cada linha, os elementos da diagonal principal devem ser maiores que a soma dos demais
    # Função retorna True quando houver convergência e False quando não é possível afirmar se há convergência
    return np.all(D > np.sum(R, axis=1))
