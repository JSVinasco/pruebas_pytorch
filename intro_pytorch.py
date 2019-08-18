# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:23:58 2019

@author: CASA
"""

###########################################################
#           un primer acercamiento a pytorch            ###
###########################################################


# funcionalidad basica estilo numpy e introduccion 'autograd' para el 
# calculo de derivadas

import torch 
import numpy as np


x = torch.ones(1, requires_grad=True)
print(x.grad)    # returns None

# un ejemplo mas elaborado de autograd

x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y* 2
z.backward()     # automatically calculates the gradient
print(x.grad)    # ∂z/∂x = 12



# primer ejemplo de una red neuronal artificial (RNA)


# a traves de la definicion de clases se definen las redes neuronales 
# en pytorch

# en el siguiente ejemplo se define un simple perceptron
# en la primera parte se define la arquitectura de la red
# que consta de una unica capa lineal
# y una funcion de activacion tipo ReLU.
# ademas en la segunda definicion se define el paso hacia 
# adelante de la red evaluando los datos en las capa definida

class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(1,1)
        self.relu = torch.nn.ReLU() # instead of Heaviside step fn
    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x) # instead of Heaviside step fn
        return output

###############################################################################
# luego para complejizar el asunto se define una red neuronal de varias capas
# usando el mismo estilo de primero definicion de la arquitectura y luego 
# la definicion de como se computan los pesos y su propagacion hacia adelante


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

# luego se pasa a la parte del entramiento de la red
# para esto se creara uno conjunto de datos sintetico en la siguiente seccion

# ademas se partiran en los subconjunto de entrenamiento y validacion

#       CREATE RANDOM DATA POINTS
            
from sklearn.datasets import make_blobs
def blob_label(y, label, loc): # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target
x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(blob_label(y_train, 0, [0]))
y_train = torch.FloatTensor(blob_label(y_train, 1, [1,2,3]))
x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0]))
y_test = torch.FloatTensor(blob_label(y_test, 1, [1,2,3]))


# una vez se tienen los datos

# se define una arquitectura pasando los parametros a la funcion 'Feedforward'
# ademas se define el criterio de perdida por medio de la funcion 'criterion'
# y la funcion de optimizacion por medio de 'optimizer'
model = Feedforward(2, 10)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# luego se define el criterio de evaluacion del modelo usando el atributo 
#'.eval()' que esta dentro del modulo torch.nn


model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before training' , before_train.item())

# a continuacion se computa el entrenamiento 


model.train()
epoch = 300
for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_train)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)
   
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()

# finalmente se evalua el modelo

model.eval()
y_pred = model(x_test)
after_train = criterion(y_pred.squeeze(), y_test) 
print('Test loss after Training' , after_train.item())
