# src/gesture_calc.py
import cv2
import mediapipe as mp
from src.utils import preprocess_drawing, predict_symbol

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
expression = ""
path_points = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[8]
        h, w, _ = frame.shape
        cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        path_points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Key Controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):  # Predict symbol
        if len(path_points) > 10:
            img = preprocess_drawing(path_points)
            symbol = predict_symbol(img)

            if symbol == '=':
                try:
                    expression = str(eval(expression))
                except:
                    expression = "Error"
            else:
                expression += symbol
        path_points = []

    elif key == ord('c'):  # Clear expression
        expression = ""
        path_points = []

    elif key == ord('q'):
        break

    # Draw current path and expression
    for i in range(1, len(path_points)):
        cv2.line(frame, path_points[i - 1], path_points[i], (255, 0, 0), 3)

    cv2.putText(frame, f"Expr: {expression}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.imshow("Gesture Calculator", frame)

cap.release()
cv2.destroyAllWindows()

