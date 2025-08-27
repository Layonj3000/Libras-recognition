import cv2
import mediapipe as mp
import numpy as np
import joblib

class HandTracker:
    """
    Classe para rastrear mãos e reconhecer gestos usando MediaPipe e um modelo de ML.
    """
    def __init__(self, model_path=None, mode=False, max_hands=1, model_complexity=1, detection_con=0.5, track_con=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.model = joblib.load(model_path) if model_path else None

    def find_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )
        return image, self.results

    def _normalizar_landmarks(self, landmarks_list):
        landmarks = np.array(landmarks_list).reshape(-1, 3)
        ponto_referencia = landmarks[0].copy()
        landmarks_relativos = landmarks - ponto_referencia
        dist_max = np.max(np.linalg.norm(landmarks_relativos, axis=1))
        if dist_max == 0:
            return np.zeros_like(landmarks_relativos.flatten())
        landmarks_normalizados = landmarks_relativos / dist_max
        return landmarks_normalizados.flatten()

    def get_gestures(self):
        """
        Analisa os landmarks e retorna uma lista de gestos preditos pelo modelo de ML,
        incluindo a confiança da predição.
        """
        detected_hands = []
        if not self.results.multi_hand_landmarks:
            return detected_hands

        for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
            # 1. Extrai e aplana os landmarks
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])

            # 2. Normaliza os dados da mesma forma que no treino
            dados_normalizados = self._normalizar_landmarks(landmarks_list)
            
            # 3. Faz a predição com o modelo
            predicao = self.model.predict([dados_normalizados])
            
            # Usamos .predict_proba() para obter a probabilidade de cada classe
            probabilidade = self.model.predict_proba([dados_normalizados])
            # A confiança é a probabilidade máxima entre todas as classes
            confianca = np.max(probabilidade)
            
            gesto_predito = predicao[0]

            # 4. Adiciona informações à lista, incluindo a confiança
            detected_hands.append({
                "index": hand_idx,
                "gesture": gesto_predito,
                "confidence": confianca, 
                "landmarks": hand_landmarks
            })
        
        return detected_hands