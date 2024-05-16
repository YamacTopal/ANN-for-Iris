# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:05:53 2024

@author: Yamaç
"""


from ucimlrepo import fetch_ucirepo 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import numpy
from sklearn.metrics import accuracy_score, precision_score, f1_score
import torch 

iris = fetch_ucirepo(id=53) 


X = iris.data.features 
y = iris.data.targets 

ct = ColumnTransformer(transformers=[('encoder', sklearn.preprocessing.OneHotEncoder(), [0])], remainder='passthrough')
y = numpy.array(ct.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = sklearn.preprocessing.StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=X.shape[1],out_features=32)
        self.fc2 = torch.nn.Linear(in_features=32, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=3)        

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x))
        
        return x
        
model = ANN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
with torch.no_grad():
    model.eval()
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5)  # Convert to binary predictions
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
        
predicted = model(X_test)
    
predicted = predicted.detach().numpy()

for i in range(len(predicted)):
    for j in range(0, 3):
        if predicted[i,j] >= 0.5:
            predicted[i, j] = 1
        else:
            predicted[i, j] = 0
    
f1 = f1_score(y_test, predicted, average='macro')
print("F1-score (Macro):", f1)
