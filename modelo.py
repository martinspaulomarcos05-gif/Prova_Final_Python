import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from dados import carregar_dados


print("Carregando dados para a classificação...")
X, y = carregar_dados()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Treinamento do modelo de Classificação (Regressão Logística)...")
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação do modelo
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print(f"\n--- Resultado de avaliação ---")
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

joblib.dump(modelo, 'modelo_steam.pkl')
print("\nModelo de classificação salvo com sucesso!")