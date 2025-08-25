import cv2
import mediapipe as mp
import csv
import os
from tqdm import tqdm 

# --- CONFIGURAÇÕES ---
DATASET_PATH = "banco_de_fotos" 
OUTPUT_CSV_FILE = 'libras_landmarks.csv'

# --- INICIALIZAÇÃO DO MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1,
    min_detection_confidence=0.5)

# --- PREPARAÇÃO DO ARQUIVO CSV ---
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

# Cria o arquivo CSV e escreve o cabeçalho se ele não existir
# Se já existir, as novas linhas serão adicionadas ao final
if not os.path.exists(OUTPUT_CSV_FILE):
    with open(OUTPUT_CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# --- PROCESSAMENTO DO DATASET ---
new_samples_count = 0
print(f"Iniciando o processamento do dataset em '{DATASET_PATH}'...")

# Pega a lista de pastas de letras (A, B, C...)
try:
    label_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
except FileNotFoundError:
    print(f"ERRO: A pasta '{DATASET_PATH}' não foi encontrada. Crie-a e organize suas fotos como instruído.")
    exit()

# Itera sobre cada pasta de letra com uma barra de progresso
for label in tqdm(label_folders, desc="Processando Letras"):
    label_path = os.path.join(DATASET_PATH, label)
    
    image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Itera sobre cada imagem dentro da pasta da letra
    for image_file in image_files:
        image_path = os.path.join(label_path, image_file)
        
        # Lê a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"\nAviso: Não foi possível ler a imagem {image_path}. Pulando.")
            continue
            
        # Converte a imagem de BGR para RGB (padrão do MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem para detectar a mão
        results = hands.process(image_rgb)
        
        # Se uma mão foi detectada, extrai e salva os landmarks
        if results.multi_hand_landmarks:
            # Pega a primeira mão detectada
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Monta a linha de dados para o CSV
            landmarks_row = [label.lower()]
            for lm in hand_landmarks.landmark:
                landmarks_row.extend([lm.x, lm.y, lm.z])
            
            # Adiciona a linha ao arquivo CSV
            with open(OUTPUT_CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks_row)
            
            new_samples_count += 1

# Finaliza o modelo do MediaPipe
hands.close()

print("\n-------------------------------------------------")
print("Processo de extração de landmarks concluído!")
print(f"{new_samples_count} novas amostras foram adicionadas ao arquivo '{OUTPUT_CSV_FILE}'.")
print("Agora você pode usar o script 'treinamento_modelo.py' para treinar seu modelo.")