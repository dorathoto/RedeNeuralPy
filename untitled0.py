# -*- coding: utf-8 -*-
"""
Criado em: Thu Apr 13 07:56:34 2023

@author: Leonardo S. Dorathoto
"""


import numpy as np
from sklearn import datasets


base = datasets.load_breast_cancer()
entradas = base.data
valoresSaida = base.target