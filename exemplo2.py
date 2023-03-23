# -*- coding: utf-8 -*-
"""
Criado em: Thu Mar  2 14:22:57 2023

@author: Leonardo S. Dorathoto
"""
import numpy as np



entradas = np.array([1, 7,5])
pesos = np.array([0.8, 0.1,0])


def soma(entrada, peso):
    return entradas.dot(peso)

s = soma(entradas, pesos)

def step_Function(soma):
    if(soma >= 1):
        return 1
    return 0

resultado = step_Function(s)