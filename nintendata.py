# Análise de Dados da série de jogos The Legend of Zelda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, hamming_loss, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

data = {
    'id': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
    'title': ['The Legend of Zelda','The Adventure of Link','A Link To The Past','Link''s Awakening','Ocarina of Time','Majora''s Mask','Oracle of Seasons','Oracle of Ages','Four Swords','The Wind Waker','Four Swords Adventures','The Minish Cap','Twilight Princess','Phantom Hourglass','Spirit Tracks','Skyward Sword','A Link Between Worlds','Hyrule Warriors','Triforce Heroes','Breath of the Wild','Cadence of Hyrule', 'Hyrule Warriors: Age of Calamity', 'Tears of the Kingdom', 'Echoes of Wisdom'],
    'year': [1987, 1988, 1992, 1993, 1998, 2000, 2001, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2009, 2011, 2013, 2014, 2015, 2017, 2019, 2020, 2023, 2024],
    'console': ['NES','NES','SNES','GameBoy','Nintendo 64','Nintendo 64','GB Color','GB Color','GB Advance','Gamecube','Gamecube','GB Advance','Wii','DS','DS','Wii','3DS','Wii U','3DS','Switch','Switch','Switch','Switch','Switch'],
    'rating': [8.4, 7.3, 9.5, 8.7, 9.9, 9.5, 8.7, 8.8, 8.5, 9.6, 8.6, 8.9, 9.5, 9.0, 8.7, 9.3, 9.1, 7.6, 7.2, 9.7, 8.5, 7.8, 9.6, 8.6],
    }

nintendata = pd.DataFrame(data)
print("---Exibição da Base de Dados---\n",nintendata)

## Verificar se a base de dados tem valores **nulos**
nintendata.isna().sum().sort_values(ascending=False)
nintendata.isnull().sum().sort_values(ascending=False)
nintendata.dropna()
nintendata.dropna(axis=1)
nintendata.drop_duplicates()

print("\n---Operações na Base de Dados---")
## Número de jogos e colunas de categoria
print("Quantidade de Linhas e Colunas: ",nintendata.shape)
print("Nome das colunas: ",nintendata.columns)

## Análise de Notas
best = nintendata[nintendata['rating'] == nintendata['rating'].max()][['title', 'rating']]
print("\nJogo com a melhor nota: \n",best)
worst = nintendata[nintendata['rating'] == nintendata['rating'].min()][['title', 'rating']]
print("Jogo com a pior nota: \n", worst)
print("Média de notas: ",nintendata['rating'].mean())
print("Variância de notas: ",nintendata['rating'].var())
print("Desvio padrão de notas: ",nintendata['rating'].std())

print("\nJogos com nota menor ou igual a 9.0: \n",nintendata[ nintendata.rating <= 9.0]) 
mean1 = len(nintendata[ nintendata.rating <= 9.0 ])/len(nintendata)
print("Percentual de jogos que possuem nota menor ou igual a 9.0: ",round(mean1*100,2),"%")

print("\nJogos com nota maior do que 9.0: \n",nintendata[ nintendata.rating >= 9.0 ])
mean2 = len(nintendata[ nintendata.rating >= 9.0 ])/len(nintendata)
print("Percentual de jogos que possuem nota menor do que 9.0: ",round(mean2*100,2),"%")

print("\nQuantidade de jogos em cada console: \n",nintendata['console'].value_counts())
print("Console com mais jogos: ",nintendata['console'].value_counts().idxmax())

print("\nSumário das operações: \n",nintendata.describe())

print("\n---Gráficos de Análise de Dados---")
sns.barplot(x=nintendata['id'], y=nintendata['rating'],hue=nintendata['console'])
plt.title("Gráfico de Barras - Notas de cada jogo")
plt.xlabel("IDs")
plt.ylabel("Notas")
plt.show()

plt.plot(nintendata['year'],nintendata['rating'],color="blue")
plt.title("Gráfico de Linha - Notas de cada jogo ao longo dos anos")
plt.xlabel("Anos")
plt.ylabel("Notas")
plt.show()

sns.histplot(nintendata['rating'],color="orange")
plt.title("Histograma - Quantos jogos tem a mesma nota?")
plt.xlabel("Notas")
plt.ylabel("Quantidade")
plt.show()

sns.boxplot(x='id',y='console',data=nintendata)
plt.title("Diagrama de Caixa - Quantos jogos tem em cada console?")
plt.xlabel("IDs")
plt.ylabel("Consoles")
plt.show()

sns.violinplot(x='id',y='console',data=nintendata)
plt.title("Diagrama de Violino - Quantos jogos tem em cada console?")
plt.xlabel("IDs")
plt.ylabel("Consoles")
plt.show()

plt.scatter(nintendata['id'],nintendata['rating'],color="green")
plt.title("Gráfico de Dispersão - Notas de cada jogo")
plt.xlabel("IDs")
plt.ylabel("Notas")
plt.show()

sns.swarmplot(x=nintendata['console'],y=nintendata['rating'],hue=nintendata['console'])
plt.title("Diagrama de Dispersão - Notas dos jogos em cada console")
plt.xlabel("Consoles")
plt.ylabel("Notas")
plt.show()

sns.kdeplot(x='rating',data=nintendata,color="red")
plt.title("Densidade de Probabilidade das Notas")
plt.xlabel("Notas")
plt.ylabel("Densidade")
plt.show()

sns.kdeplot(x='id',y='rating',data=nintendata, color="purple")
plt.title("Regressão Linear das Notas")
plt.xlabel("Notas")
plt.ylabel("Densidade")
plt.show()

print("\n---Machine Learning---")
print("-Regressão Linear-")
nintendata['console_num'] = LabelEncoder().fit_transform(nintendata['console'])
x = nintendata[['year','console_num']]
y = nintendata['rating']
train1, test1, train2, test2 = train_test_split(x, y, test_size = 0.2, random_state = 42) #Treinos e testes

linear_model = LinearRegression()
linear_model.fit(train1, train2) #Criação do modelo de regressão linear

preview2 = linear_model.predict(test1) #Previsão do modelo

print("Erro médio quadrado do modelo: ",mean_squared_error(test2, preview2))
print("Coeficiente de determinação do modelo: ",r2_score(test2, preview2))

plt.scatter(test2, preview2)
plt.xlabel("Valores reais")
plt.ylabel("Valores previstos")
plt.title("Gráfico de Dispersão - Valores Reais vs. Valores Previstos")
plt.show()

print("\n-Regressão Logística-")
nintendata['consolenum'] = LabelEncoder().fit_transform(nintendata['console'])
nintendata['rating_category'] = pd.cut(nintendata['rating'], bins=(0,6.0,8.0,10.0), labels=['Baixo','Médio','Alto'])
x = nintendata[['year','consolenum']]
y = nintendata['rating_category']
train3, test3, train4, test4 = train_test_split(x, y, test_size = 0.2, random_state = 42) #Treinos e testes

logistic_model = LogisticRegression()
logistic_model.fit(train3, train4) #Criação do modelo de regressão linear

preview4 = logistic_model.predict(test3) #Previsão do modelo

accuracy = accuracy_score(test4, preview4)
print("Acurácia do modelo: ",accuracy * 100,"%")

mat = confusion_matrix(test4, preview4)
print("Matriz de confusão: \n",mat)

report = classification_report(test4, preview4)
print("Relatório de classificação do modelo: \n",report)

print("Código de Hamming: ",hamming_loss(test4,preview4))

sns.heatmap(mat, annot=True, cmap='Greens')
plt.title("Matriz de Confusão")
plt.xlabel("Valores previstos")
plt.ylabel("Valores reais")
plt.show()

train3 = StandardScaler().fit_transform(train3)
test3 = StandardScaler().fit_transform(test3) #Padronização dos dados

cross_val_score(logistic_model, x, y, cv=10) #Validação cruzada

print("\n-Árvore de Decisão-")
dt_model = tree.DecisionTreeClassifier(max_depth=5)
dt_model.fit(train3, train4) #Criação do modelo de árvore de decisão

dtpreview4 = dt_model.predict(test3) #Previsão do modelo de árvore de decisão

accuracy = accuracy_score(test4, dtpreview4) #Acurácia do modelo de árvore de decisão
print("Acurácia do modelo: ",accuracy * 100,"%")

plt.figure(figsize=(8,4))
tree.plot_tree(dt_model, filled=True, feature_names=['year','consolenum'])
plt.show() #Gráfico da árvore de decisão

print("\n-Árvores Aleatórias-")
rfl_model = RandomForestClassifier(n_estimators=100, random_state=50)
rfl_model.fit(train3, train4) #Criação de modelo de árvores aleatórias

rflpreview4 = rfl_model.predict(test3) #Previsão do modelo de árvores aleatórias

accuracy = accuracy_score(test4, rflpreview4) #Acurácia do modelo de árvores aleatórias
print("Acurácia do modelo: ",accuracy * 100,"%")

feature = rfl_model.feature_importances_ #Importância das colunas
print("Importância das colunas: ",feature * 100)
