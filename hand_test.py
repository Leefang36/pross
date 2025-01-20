import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Mediapipe Hands and Drawing utilities
hands_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Smoothing variables   
alpha = 0.85
prev_mouse_x, prev_mouse_y = 0, 0

# Cursor speed adjustment
speed_factor = 1.5

# Scrolling speed factor
scroll_speed = 55

# Delay timer
last_action_time = time.time()
action_delay = 0.1  # Delay between different actions

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands with Mediapipe
    results = hands_detector.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Get landmark coordinates
            landmarks = hand_landmarks.landmark

            # Index finger (ID 8) and middle finger (ID 12)
            index_tip = landmarks[8]
            middle_tip = landmarks[12]

            # Map index finger position to screen coordinates
            index_x = int(index_tip.x * frame.shape[1])
            index_y = int(index_tip.y * frame.shape[0])
            mouse_x = int(alpha * prev_mouse_x + (1 - alpha) * (index_tip.x * screen_width * speed_factor))
            mouse_y = int(alpha * prev_mouse_y + (1 - alpha) * (index_tip.y * screen_height * speed_factor))

            # Update cursor position
            pyautogui.moveTo(mouse_x, mouse_y)
            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

            # Draw circle on the index tip
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

            # Detect click gesture based on index and middle finger proximity
            middle_x = int(middle_tip.x * frame.shape[1])
            middle_y = int(middle_tip.y * frame.shape[0])
            dist = np.sqrt((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2)

            if dist < 30 and time.time() - last_action_time > action_delay:  # Adjust threshold as needed
                cv2.putText(frame, "Click!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.click()
                last_action_time = time.time()

            # Scrolling gesture: Index finger up, others closed
            ring_y = int(landmarks[16].y * frame.shape[0])
            pinky_y = int(landmarks[20].y * frame.shape[0])
            middle_y = int(landmarks[12].y * frame.shape[0])

            if (index_y < ring_y - 30 and index_y < pinky_y - 30 and
                index_y < middle_y - 30 and time.time() - last_action_time > action_delay):
                pyautogui.scroll(scroll_speed)  # Scroll up
                cv2.putText(frame, "Scroll Up", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_action_time = time.time()
            elif (ring_y < index_y - 30 and pinky_y < index_y - 30 and
                  middle_y < index_y - 30 and time.time() - last_action_time > action_delay):
                pyautogui.scroll(-scroll_speed)  # Scroll down
                cv2.putText(frame, "Scroll Down", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                last_action_time = time.time()

            # Navigate back in browser: Index finger pointing left
            thumb_tip = landmarks[4]
            thumb_x = int(thumb_tip.x * frame.shape[1])
            folded_fingers = all(landmarks[f].y > landmarks[f - 2].y for f in [12, 16, 20])

            if folded_fingers and landmarks[8].x < 0.5 and time.time() - last_action_time > action_delay:
                cv2.putText(frame, "Back", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                pyautogui.hotkey('alt', 'left')
                last_action_time = time.time()

            # Navigate forward in browser: Thumb pointing left, all fingers folded
            if folded_fingers and thumb_x < frame.shape[1] // 2 and thumb_tip.y < landmarks[3].y and time.time() - last_action_time > action_delay:
                cv2.putText(frame, "Forward", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pyautogui.hotkey('alt', 'right')
                last_action_time = time.time()

    # Add instructions to the frame
    cv2.putText(frame, "Press ESC to Exit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Mouse Control", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
