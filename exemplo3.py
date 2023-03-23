# -*- coding: utf-8 -*-
"""
Criado em: Fri Mar  3 07:22:45 2023

@author: Leonardo S. Dorathoto

Perceptron operador lógico AND

Perceptron é uma rede neural de camada única.
O Perceptron é um classificador linear (binário). 
Além disso, é usado na aprendizagem supervisionada e pode ser usado para classificar os dados de entrada fornecidos.

x1 * peso1 + x2 * peso2 = Valor step
"""

import numpy as np

entradas = np.array([[0,0],[0,1],[1,0],[1,1] ])
saidas = np.array([0,0,0,1])
pesos = np.array([0.0, 0.0])
#pesos = np.zeros((2, 1))
taxaAprendizado = 0.1


def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0


def calculaSaida(registro):
    soma = registro.dot(pesos)
    return stepFunction(soma)

        
def treinar():
    erroTotal = 1
    epoca=1
    while(erroTotal != 0):
        erroTotal = 0
        print("\n\n\nTREINAMENTO \t\t{ind}".format(ind=epoca))
        epoca += 1
        for i in range(len(saidas)): 
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):           
                pesos[j] = pesos[j] + (taxaAprendizado * entradas[i][j] * erro)
                print('Peso atualizado: ' + str(pesos[j]))
        print('Total de erros: \t{erros} '.format(erros=erroTotal))
    print('Epoca.: \t\t\t' + str(epoca))        
treinar()
print("Rede neural treinada")
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))
                