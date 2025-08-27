import cv2
import csv
import os
from hand_tracker import HandTracker  

# --- CONFIGURAÇÕES ---
NOME_ARQUIVO_CSV = 'libras_landmarks.csv'
LETRAS_PARA_COLETAR = "abcdefgilmnopqrstuvwy" # Tirei as letras que tem movimentos grandes(h, j, k, x, z)

# --- INICIALIZAÇÃO DO HANDTRACKER ---
tracker = HandTracker(model_path=None, max_hands=1)

# --- PREPARAÇÃO DO ARQUIVO CSV ---
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

if not os.path.exists(NOME_ARQUIVO_CSV):
    with open(NOME_ARQUIVO_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# --- CAPTURA DE VÍDEO ---
cap = cv2.VideoCapture(0)

print(">>> INSTRUÇÕES <<<")
print("Faça o sinal da letra desejada e pressione a tecla correspondente no teclado.")
print(f"Letras a serem coletadas: {LETRAS_PARA_COLETAR}")
print("Pressione 'q' para sair.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    image = cv2.flip(image, 1)
    image, results = tracker.find_hands(image, draw=True)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            key = cv2.waitKey(5) & 0xFF
            char_key = chr(key).lower()

            if char_key in LETRAS_PARA_COLETAR:
                landmarks_row = [char_key]
                for lm in hand_landmarks.landmark:
                    landmarks_row.extend([lm.x, lm.y, lm.z])

                with open(NOME_ARQUIVO_CSV, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks_row)

                print(f"Dados salvos para a letra: '{char_key.upper()}'")

    cv2.imshow('Coleta de Dados - LIBRAS', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Coleta finalizada. Dados salvos em '{NOME_ARQUIVO_CSV}'.")
