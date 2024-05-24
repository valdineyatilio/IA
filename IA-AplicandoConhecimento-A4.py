#Aplicando Conhecimento - A4 
# Nome: Valdiney Atílio Pedro
#Ra: 10424616
#Patrícia Corrêa França
#Ra: 10423533
#Mariana Simoes Rubio
#Ra:10424388
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
train_data = pd.read_csv('https://github.com/valdineyatilio/IA/raw/main/Introdu%C3%A7%C3%A3o%20%C3%A0%20Intelig%C3%AAncia%20Artificial%20-%20Aula%204%20-%20Base%20de%20Dados%20Atividade/iris-train.data')
test_data = pd.read_csv('https://github.com/valdineyatilio/IA/raw/main/Introdu%C3%A7%C3%A3o%20%C3%A0%20Intelig%C3%AAncia%20Artificial%20-%20Aula%204%20-%20Base%20de%20Dados%20Atividade/iris-test.data')

# Preparar os dados
features_train = train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
labels_train = train_data['species']

features_test = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
labels_test = test_data['species']

le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_test = le.transform(labels_test)

# Converter para tensores PyTorch
featuresTrain = torch.tensor(features_train.values).float()
labelsTrain = torch.tensor(labels_train).long()

featuresTest = torch.tensor(features_test.values).float()
labelsTest = torch.tensor(labels_test).long()

# Definir a rede MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        
    def forward(self, x):
        return self.layers(x)

# Instanciar a rede e definir o otimizador e a função de perda
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Treinar a rede
for epoch in range(100):
    optimizer.zero_grad()
    output = model(featuresTrain)
    loss = loss_fn(output, labelsTrain)
    loss.backward()
    optimizer.step()

# Avaliar a rede
model.eval()
with torch.no_grad():
    output = model(featuresTest)
    _, predicted = torch.max(output, 1)
    print('Acurácia: ', (predicted == labelsTest).sum().item() / len(labelsTest))
