#
#   Herramientas recopiladas del curso de "Deep Learning: Mastering neural networks" del MIT (Enero 2023)
#

import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import sklearn.metrics as metrics
from torch.utils.data import TensorDataset, DataLoader 
import torchvision.transforms as transforms
import time, copy
import torchvision



# feature_columns:  List(String)    - list of column names to be used as features (X)
# label_column:     String          - column to be used as target (Y)
# ct:               ColumnTransformer
# Returns:          TensorDataset   - train, test, val 
def train_test_val_split(df, feature_columns, label_column, ct):
    
    # Primero, realizar una división 80/20 en entrenamiento y prueba
    seed = 42
    initial_train_split = df.sample(frac=.8, random_state = seed)
    test = df.drop(initial_train_split.index)   # remover todo este subset del dataset original
    # test = 20%
    # init = 80%

    # A continuación, realizar una división 75/25 entre entrenamiento y validación
    train = initial_train_split.sample(frac=.75, random_state = seed)
    val = initial_train_split.drop(train.index)
    # val   = 25%
    # train = 75%

    # Dividir todos los subconjuntos en atributos y etiquetas (x e y)
    train_x = train[feature_columns]
    # No transformaremos las etiquetas para que vayan directamente a los tensores torch
    train_y = torch.from_numpy(train[label_column].values)

    val_x = val[feature_columns]
    val_y = torch.from_numpy(val[label_column].values)

    test_x = test[feature_columns]
    test_y = torch.from_numpy(test[label_column].values)

    # Ajustar ColumnTransfer al conjunto de entrenamiento
    ct.fit(train_x)

    # Realizar la estandarización con cada uno de los conjuntos de datos x
    train_x = ct.transform(train_x)
    val_x = ct.transform(val_x)
    test_x = ct.transform(test_x)

    # Convertir los conjuntos de datos en tensores Torch
    train_x = torch.from_numpy(train_x).float()
    val_x = torch.from_numpy(val_x).float()
    test_x = torch.from_numpy(test_x).float()

    # Crear los conjuntos de datos de pares input-etiqueta para que PyTorch los consuma
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    val_dataset = TensorDataset(val_x, val_y)
    return train_dataset, test_dataset, val_dataset