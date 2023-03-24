# -*- coding: utf-8 -*-
"""
XOR


Criado em: Thu Mar 16 08:00:40 2023
@author: Leonardo S. Dorathoto
atributos previsores (entrada) aula22

gradiente cost function (min local, min global)
"""

import numpy as np




def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

#a = sigmoid(0.5)
#b = sigmoidDerivada(0.406)

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

#saidasTest = np.array([0,1,1,0])
saidas = np.array([[0],[1],[1],[0]])

#camada oculta
#pesos0 = np.array([[-0.424,-0.740,-0.961],[0.358, -0.577,-0.469]])
pesos0 = 2* np.random.random((2,3))-1

#camada saida
#pesos1 = np.array([[-0.017],[-0.893],[0.148]])
pesos1 = np.random.random((3,1))-1

epocas = 10000
txAprendizagem = 0.6
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    #np.mean (média aritmética)
    #np.abs (número absoluto (positivo))
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    
    print(str(j) + " - Erro: " + str(mediaAbsoluta) + " - Acerto: " + str((1 - mediaAbsoluta)* 100))

    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    
    #matriz transposta
    pesos1Transposta = pesos1.T
    deltaSaidaXPesos = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPesos * sigmoidDerivada(camadaOculta)
    
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovos1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovos1 * txAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * txAprendizagem)
    