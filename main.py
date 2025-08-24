# Arquivo: main.py

import cv2
from hand_tracker import HandTracker

def main():
    """
    Função principal que usa a classe HandTracker para detecção em tempo real.
    """
    cap = cv2.VideoCapture(0)
    
    tracker = HandTracker(model_path='modelo_libras.pkl', max_hands=1)
    
    CONFIDENCE_THRESHOLD = 0.50

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        image = cv2.flip(image, 1)
        image, _ = tracker.find_hands(image)
        detected_hands = tracker.get_gestures()

        if detected_hands:
            hand = detected_hands[0]
            gesto = hand["gesture"]
            confianca = hand["confidence"] 
            landmarks = hand["landmarks"]
            
            if confianca > CONFIDENCE_THRESHOLD:
                texto_resultado = f"{gesto.upper()} ({confianca * 100:.1f}%)"
                
                wrist = landmarks.landmark[0]
                h, w, _ = image.shape
                pos_x = int(wrist.x * w) - 50
                pos_y = int(wrist.y * h) - 40

                cv2.putText(image, texto_resultado, (pos_x, pos_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Detecção de LIBRAS com Machine Learning', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()