import cv2
import csv
import os
from tqdm import tqdm
from hand_tracker import HandTracker 

# --- CONFIGURAÇÕES ---
DATASET_PATH = "banco_de_fotos"
OUTPUT_CSV_FILE = 'libras_landmarks.csv'

# --- INICIALIZAÇÃO DO HANDTRACKER ---
tracker = HandTracker(model_path=None, mode=True, max_hands=1)

# --- PREPARAÇÃO DO ARQUIVO CSV ---
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

if not os.path.exists(OUTPUT_CSV_FILE):
    with open(OUTPUT_CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# --- PROCESSAMENTO DO DATASET ---
new_samples_count = 0
print(f"Iniciando o processamento do dataset em '{DATASET_PATH}'...")

try:
    label_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
except FileNotFoundError:
    print(f"ERRO: A pasta '{DATASET_PATH}' não foi encontrada. Crie-a e organize suas fotos como instruído.")
    exit()

# Itera sobre cada pasta de letra
for label in tqdm(label_folders, desc="Processando Letras"):
    label_path = os.path.join(DATASET_PATH, label)
    image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for image_file in image_files:
        image_path = os.path.join(label_path, image_file)

        image = cv2.imread(image_path)
        if image is None:
            print(f"\nAviso: Não foi possível ler a imagem {image_path}. Pulando.")
            continue

        # Usa o HandTracker para processar a imagem
        _, results = tracker.find_hands(image, draw=False)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            landmarks_row = [label.lower()]
            for lm in hand_landmarks.landmark:
                landmarks_row.extend([lm.x, lm.y, lm.z])

            with open(OUTPUT_CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks_row)

            new_samples_count += 1

print("\n-------------------------------------------------")
print("Processo de extração de landmarks concluído!")
print(f"{new_samples_count} novas amostras foram adicionadas ao arquivo '{OUTPUT_CSV_FILE}'.")
print("Agora você pode usar o script 'treinamento_modelo.py' para treinar seu modelo.")