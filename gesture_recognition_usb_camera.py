import cv2
from PIL import Image
import numpy as np
import mediapipe as mp
import pyautogui
import webbrowser
import time

# Load the Mediapipe HandPose model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def draw_hand_landmarks(frame_rgb, hand_landmarks):
    for index, landmark in enumerate(hand_landmarks.landmark):
        x, y = int(landmark.x * frame_rgb.shape[1]), int(landmark.y * frame_rgb.shape[0])
        cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_rgb, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return frame_rgb

def create_hand_mask(frame, hand_landmarks, padding=20):
    # Create an empty mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Get the coordinates for the landmarks
    landmark_coords = [(int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])) for landmark in hand_landmarks.landmark]

    # Create a convex hull around the hand landmarks
    hull = cv2.convexHull(np.array(landmark_coords))
    
    # Apply padding to the hull
    hull_padded = cv2.erode(cv2.dilate(hull, None, iterations=padding), None, iterations=padding)
    
    # Fill the convex hull to create the mask
    cv2.fillConvexPoly(mask, hull_padded, 255)

    return mask

def process_hand_gestures(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply Gaussian blur
    blur_size = (5, 5)
    frame_rgb_blurred = cv2.GaussianBlur(frame_rgb, blur_size, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_rgb_blurred, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding
    max_output_value = 255
    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # Can also use ADAPTIVE_THRESH_MEAN_C
    threshold_type = cv2.THRESH_BINARY
    block_size = 11  # Size of a pixel neighborhood that is used to calculate a threshold value
    c = 2  # A constant subtracted from the mean or weighted mean

    thresholded = cv2.adaptiveThreshold(gray, max_output_value, adaptive_method, threshold_type, block_size, c)

    # Convert back to color so we can draw colored landmarks on it
    frame_rgb_thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

    result = hands.process(frame_rgb_thresholded)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Create a binary mask for the detected hand
            hand_mask = create_hand_mask(frame_rgb_thresholded, hand_landmarks)
           
            # Apply the mask to the frame
            frame_rgb_masked = cv2.bitwise_and(frame_rgb_thresholded, frame_rgb_thresholded, mask=hand_mask)
            frame_rgb_thresholded = draw_hand_landmarks(frame_rgb_thresholded, hand_landmarks)
            perform_actions_based_on_gestures(hand_landmarks)

    return frame_rgb_thresholded



def is_pointer_finger_up(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    is_finger_above_wrist = index_finger_tip.y < wrist.y
    is_finger_above_thumb_mcp = index_finger_tip.y < thumb_mcp.y

    return is_finger_above_wrist and is_finger_above_thumb_mcp


def enhance_image_contrast(image):
    # Define constants
    THRESHOLD = 255
    ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_MEAN_C
    BLOCK_SIZE = 11
    C = 2
    KERNEL_SIZE = (3, 3)
    BLUR_SIZE = (3, 3)
    CLIP_LIMIT = 2.0
    TILE_GRID_SIZE = (8, 8)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance image contrast
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    enhanced = clahe.apply(gray)

    # Apply adaptive thresholding to binarize the image
    threshold = cv2.adaptiveThreshold(enhanced, THRESHOLD, ADAPTIVE_METHOD, cv2.THRESH_BINARY, BLOCK_SIZE, C)

    # Apply morphological opening to remove noise
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply Gaussian blur to smooth the image
    blur = cv2.GaussianBlur(opening, BLUR_SIZE, 0)
    
    return blur

def perform_actions_based_on_gestures(hand_landmarks):
    volume_steps = 5  # Adjust this value to change the number of volume steps

    if is_thumbs_up(hand_landmarks):
        print("Thumbs up detected!")
        for _ in range(volume_steps):
            pyautogui.press('volumeup')
    elif is_thumbs_down(hand_landmarks):
        print("Thumbs down detected!")
        for _ in range(volume_steps):
            pyautogui.press('volumedown')
    elif are_all_fingers_outstretched(hand_landmarks):
        print("All fingers outstretched detected!")
        pyautogui.hotkey('win', 'd')
    elif is_longhorn(hand_landmarks):
        print("BEVO detected!")
        webbrowser.open_new_tab("https://www.youtube.com/watch?v=W41tB1nkQtI")



def are_all_fingers_outstretched(hand_landmarks):
    if hand_landmarks is None:
        return False

    thumb_outstretched = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].x

    other_fingers_outstretched = all(
        [
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y <
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].y
        ]
    )
    return thumb_outstretched and other_fingers_outstretched

def is_thumbs_up(hand_landmarks):
    if hand_landmarks is None:
        return False
    thumb_outstretched = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].y

    other_fingers_folded = all(
            [   hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x >
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].x
            ]
        )
        
    return thumb_outstretched and other_fingers_folded
    
def is_thumbs_down(hand_landmarks):
    if hand_landmarks is None:
        return False
    thumb_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].y
    
    other_fingers_folded = all(
            [        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x <
                    hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].x,
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x <
                hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].x
            ]
    )
        
    return thumb_folded and other_fingers_folded


def is_longhorn(hand_landmarks):
    if hand_landmarks is None:
        return False

    # Index and pinky fingers are extended
    index_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y
    pinky_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y
    fingers_extended = (index_tip_y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP].y) and \
                       (pinky_tip_y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP].y)

    # Middle two fingers are folded into the palm
    middle_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y
    ring_tip_y = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y
    fingers_folded = (middle_tip_y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y) and \
                     (ring_tip_y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP].y)

    # Thumb can be extended or folded
    thumb_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x
    thumb_extended = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].y

    # Return True if all conditions are met
    return fingers_extended and fingers_folded and (thumb_folded or thumb_extended)
        
def main():
    cap = cv2.VideoCapture(0)

    gesture_recognized = False
    gesture_time = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Unable to read the video feed.")
            break

        # Check if the gesture recognition delay has passed
        if not gesture_recognized or time.time() - gesture_time >= 1.5:
            frame_rgb = process_hand_gestures(frame, hands)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Set gesture_recognized to True and store the time
            gesture_recognized = True
            gesture_time = time.time()
        else:
            frame_bgr = frame
            
        # Resize the frame
        scale_factor = 2
        new_width = int(frame_bgr.shape[1] * scale_factor)
        new_height = int(frame_bgr.shape[0] * scale_factor)
        frame_resized = cv2.resize(frame_bgr, (new_width, new_height))
        
        cv2.imshow("Hand Gestures", frame_resized)

        # Press space bar to exit
        if cv2.waitKey(1) & 0xFF == 32:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()
