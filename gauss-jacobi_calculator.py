import numpy as np
from scipy.linalg import solve
from functions import jacobi, convergence

def main():
    """
    Resolução de Sistemas de Equações Lineares
    pelo Método Iterativo de Jacobi
    """
    
    opcao = 0
    while(opcao != 1):
        A = np.array(np.matrix(input(f"\nInsira a Matriz dos Coeficientes A (ex.: a11, a12, ..., a1n; ...; ann, ...): "), dtype=float))

        if(convergence(A)):
            print(f"\nCriterio das Linhas satisfeito, o sistema possui convergencia")
            break;
        else:
            print(f"\nCriterio das Linhas nao foi satisfeito, nao e possivel afirmar que o sistema possui convergencia")
            print(f"\nMenu de opcoes:")
            print("  0 - Inserir a Matriz dos Coeficientes A novamente permutando linhas/colunas")
            print("  1 - Continuar de qualquer forma")
            opcao = int(input())
            
    b = np.fromstring(input(f"\nInsira a Matriz dos Termos Independentes b (ex.: b1, b2, ..., bn): "), dtype=float, sep=',')
    E = float(input(f"\nInsira o erro desejado: "))
    N = int(input(f"\nInsira o numero maximo de iteracoes desejado: "))
    guess = np.fromstring(input(f"\nInsira a Matriz Inicial de chute (ex.: x01, x02, ..., x0n): "), dtype=float, sep=',')
    
    print(f"\nMatriz dos Coeficientes A: \n{A}")
    print(f"\nMatriz dos Termos Independentes b: \n{np.vstack(b)}")

    solucao = jacobi(A, b, E, N, guess)
    print(f"\nMatrix Solucao Aproximada x: \n{np.vstack(solucao[0])}")
    print(f"\nErro: {solucao[1]}")
    print(f"Numero de Iteracoes: {solucao[2]}")

    print(f"\nSolucao Real: \n{np.vstack(solve(A, b))}")

if __name__ == "__main__":
    main()
