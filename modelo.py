import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from dados import carregar_dados


print("Carregando dados para a classificação...")
X, y = carregar_dados()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"\n--- Estatísticas do Conjunto de Teste ---")
print(f"Tamanho do teste: {len(X_test)} exemplos")
print("Distribuição das classes reais no teste (%):")
print(y_test.value_counts(normalize=True) * 100)

# Calcula o baseline (proporção da classe mais frequente)
baseline = y_test.value_counts(normalize=True).max() * 100
print(f"Baseline (Classe mais frequente): {baseline:.2f}%")


print("\n--- Treinando e Avaliando Modelos ---")


# 1. Primeiro Modelo: Regressão Logística
modelo_logistico = LogisticRegression(max_iter=1000, random_state=42)
modelo_logistico.fit(X_train, y_train)
pred_logistico = modelo_logistico.predict(X_test)
acc_logistico = accuracy_score(y_test, pred_logistico)
print(f"Acurácia da Regressão Logística: {acc_logistico * 100:.2f}%")
print("Matriz de Confusão (Regressão Logística):")
print(confusion_matrix(y_test, pred_logistico))

# 2. Segundo Modelo: Árvore de Decisão
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train, y_train)
pred_arvore = modelo_arvore.predict(X_test)
acc_arvore = accuracy_score(y_test, pred_arvore)
print(f"Acurácia da Árvore de Decisão: {acc_arvore * 100:.2f}%")
print("Matriz de Confusão (Árvore de Decisão):")
print(confusion_matrix(y_test, pred_arvore))

melhor_modelo = modelo_logistico if acc_logistico >= acc_arvore else modelo_arvore
nome_escolhido = "Regressão Logística" if melhor_modelo == modelo_logistico else "Árvore de Decisão"

print(f"\nModelo escolhido para salvar e utilizar no sistema: {nome_escolhido}")

# Salva o arquivo binário que o app.py vai ler
joblib.dump(melhor_modelo, 'modelo_steam.pkl')
print("Modelo salvo com sucesso como 'modelo_steam.pkl'!")