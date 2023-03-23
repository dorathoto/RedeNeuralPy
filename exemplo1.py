# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:11:21 2023

@author: Leonardo
"""



entradas = [1,7,5]
pesos = [0.8,0.1,0]


def soma(entrada, peso):
    soma = 0
    for i in range(3):
        #print(entradas[i])
        #print(pesos[i])
        soma += entrada[i] * peso[i]
    return soma

s = soma(entradas, pesos)

def step_Function(soma):
    if(soma >= 1):
        return 1
    return 0

resultado = step_Function(s)