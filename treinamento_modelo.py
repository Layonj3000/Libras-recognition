import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

NOME_ARQUIVO_CSV = 'libras_landmarks.csv'
NOME_ARQUIVO_MODELO = 'modelo_libras.pkl'

# --- CARREGAMENTO DOS DADOS ---
print(f"Carregando dados de '{NOME_ARQUIVO_CSV}'...")
df = pd.read_csv(NOME_ARQUIVO_CSV)
X = df.drop('label', axis=1)
y = df['label']
print(f"Total de {len(df)} amostras carregadas.")
print("Distribuição das classes:")
print(y.value_counts().sort_index())

# --- PRÉ-PROCESSAMENTO E NORMALIZAÇÃO ---
def normalizar_landmarks(landmarks_row):
    landmarks = np.array(landmarks_row).reshape(-1, 3)
    ponto_referencia = landmarks[0].copy()
    landmarks_relativos = landmarks - ponto_referencia
    dist_max = np.max(np.linalg.norm(landmarks_relativos, axis=1))
    if dist_max == 0:
        return np.zeros_like(landmarks_relativos.flatten())
    landmarks_normalizados = landmarks_relativos / dist_max
    return landmarks_normalizados.flatten()

print("\nProcessando e normalizando os dados...")
X_processed = X.apply(normalizar_landmarks, axis=1, result_type='expand')

# --- DIVISÃO EM TREINO E TESTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDados divididos em {len(X_train)} amostras de treino e {len(X_test)} de teste.")

# --- TREINAMENTO DO MODELO ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTreinando o modelo RandomForestClassifier...")
model.fit(X_train, y_train)
print("Treinamento concluído.")

# --- AVALIAÇÃO DO MODELO ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia no conjunto de teste: {accuracy * 100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- SALVANDO O MODELO TREINADO ---
joblib.dump(model, NOME_ARQUIVO_MODELO)
print(f"\nModelo salvo com sucesso em '{NOME_ARQUIVO_MODELO}'.")