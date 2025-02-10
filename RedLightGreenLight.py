# import cv2
# import mediapipe as mp
# import pygame
# import os
# import time

# # Initialize Pygame for sound playback
# pygame.init()

# # File paths for game visuals and sounds
# folderPath = 'frames'
# sounds = {
#     "intro": "sounds/squidWin.mp3",
#     "kill": "sounds/kill.mp3",
#     "win": "sounds/win.mp3",
#     "green_light": "sounds/green.mp3",
#     "red_light": "sounds/red.mp3"
# }

# # Load frames for the game visuals
# mylist = os.listdir(folderPath)
# graphic = [cv2.imread(f'{folderPath}/{imPath}') for imPath in sorted(mylist)]
# green = graphic[0]
# red = graphic[1]
# kill = graphic[2]  # Blood/elimination frame
# winner = graphic[3]
# intro = graphic[4]

# # MediaPipe Pose setup
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Game timer and movement detection setup
# TIMER_MAX = 20  # Increased to 20 seconds
# TIMER = TIMER_MAX
# motion_threshold = 0.30  # Motion threshold for both players
# font = cv2.FONT_HERSHEY_SIMPLEX
# cap = cv2.VideoCapture(0)
# is_green_light = True
# prev_landmarks_p1, prev_landmarks_p2 = None, None  # To store previous pose landmarks for players
# player_lost = [False, False]  # Track whether player 1 or player 2 lost

# # Show the intro screen and play intro sound
# cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.5, fy=0.5))
# pygame.mixer.music.load(sounds["intro"])
# pygame.mixer.music.play()
# cv2.waitKey(2000)  # Wait for intro sound to finish

# prev_time = time.time()
# win = False

# def calculate_motion(landmarks1, landmarks2):
#     """Calculate motion by summing up the distance between corresponding landmarks."""
#     if landmarks1 is None or landmarks2 is None:
#         return 0

#     motion = sum(
#         ((landmarks1[i].x - landmarks2[i].x) ** 2 + (landmarks1[i].y - landmarks2[i].y) ** 2) ** 0.5
#         for i in range(len(landmarks1))
#         if landmarks1[i].visibility > 0.5 and landmarks2[i].visibility > 0.5
#     )
#     return motion

# try:
#     while cap.isOpened() and TIMER >= 0:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame.")
#             break

#         frame = cv2.flip(frame, 1)  # Flip horizontally for a natural mirror effect
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = pose.process(rgb_frame)

#         # Draw landmarks on the frame
#         if result.pose_landmarks:
#             mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         # Display the game timer
#         show_frame = cv2.resize(green if is_green_light else red, (0, 0), fx=0.5, fy=0.5)
#         cv2.putText(show_frame, f'Timer: {TIMER}', (20, 50), font, 1, (0, 0, 255), 2)

#         # Check if time to switch light
#         cur_time = time.time()
#         if cur_time - prev_time >= 1:
#             prev_time = cur_time
#             TIMER -= 1
#             is_green_light = not is_green_light  # Switch light

#             # Play corresponding light sound
#             if is_green_light:
#                 pygame.mixer.music.load(sounds["green_light"])
#             else:
#                 pygame.mixer.music.load(sounds["red_light"])
#             pygame.mixer.music.play()

#             # Store previous landmarks for motion detection
#             prev_landmarks_p1 = result.pose_landmarks.landmark if result.pose_landmarks else None
#             prev_landmarks_p2 = prev_landmarks_p1  # Simulate second player (for webcam simplification)

#         # Detect movement during red light
#         if not is_green_light and result.pose_landmarks:
#             motion_p1 = calculate_motion(prev_landmarks_p1, result.pose_landmarks.landmark)
#             motion_p2 = calculate_motion(prev_landmarks_p2, result.pose_landmarks.landmark)
#             print(f"Detected motion - Player 1: {motion_p1}, Player 2: {motion_p2}")  # For debugging

#             # Check motion thresholds for both players
#             if motion_p1 > motion_threshold:
#                 player_lost[0] = True  # Player 1 lost
#             if motion_p2 > motion_threshold:
#                 player_lost[1] = True  # Player 2 lost

#             # If any player loses, show the kill frame and play sound
#             if any(player_lost):
#                 pygame.mixer.music.load(sounds["kill"])
#                 pygame.mixer.music.play()
#                 for _ in range(60):  # Display kill frame for ~3 seconds
#                     cv2.imshow("Squid Game", cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
#                     cv2.waitKey(50)
#                 break  # End the game if any player loses

#         # Display the current frame and webcam view
#         cam_show = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
#         cam_h, cam_w = cam_show.shape[:2]
#         show_frame[0:cam_h, -cam_w:] = cam_show  # Place webcam in the top right corner
#         cv2.imshow("Squid Game", show_frame)

#         # Win condition
#         if TIMER == 0 and is_green_light:
#             win = True
#             break

#         # Handle quit event
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     # After game logic: Win or lose scenario
#     if win:
#         pygame.mixer.music.load(sounds["win"])
#         pygame.mixer.music.play()
#         for _ in range(60):  # Display winner frame for 3 seconds
#             cv2.imshow("Squid Game", cv2.resize(winner, (0, 0), fx=0.5, fy=0.5))
#             cv2.waitKey(50)
#     elif any(player_lost):
#         print("Player eliminated!")  # For debugging purposes
#     else:
#         # If the player runs out of time but didn't move during red light
#         pygame.mixer.music.load(sounds["kill"])
#         pygame.mixer.music.play()
#         for _ in range(60):  # Display kill frame for 3 seconds
#             cv2.imshow("Squid Game", cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
#             cv2.waitKey(50)

# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#     pygame.quit()

import cv2
import mediapipe as mp
import pygame
import os
import time
import random  # For random delay

# Initialize Pygame for sound playback
pygame.init()

# File paths for game visuals and sounds
folderPath = 'frames'
sounds = {
    "intro": "sounds/squidWin.mp3",
    "kill": "sounds/kill.mp3",
    "win": "sounds/win.mp3",
    "green_light": "sounds/green.mp3",
    "red_light": "sounds/red.mp3"
}

# Load frames for the game visuals
mylist = os.listdir(folderPath)
graphic = [cv2.imread(f'{folderPath}/{imPath}') for imPath in sorted(mylist)]
green = graphic[0]
red = graphic[1]
kill = graphic[2]  # Blood/elimination frame
winner = graphic[3]
intro = graphic[4]

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Game timer and movement detection setup
TIMER_MAX = 20  # Increased to 20 seconds
TIMER = TIMER_MAX
motion_threshold = 0.30  # Motion threshold for both players
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
is_green_light = True
prev_landmarks_p1, prev_landmarks_p2 = None, None  # To store previous pose landmarks for players
player_lost = [False, False]  # Track whether player 1 or player 2 lost

# Show the intro screen and play intro sound
cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.5, fy=0.5))
pygame.mixer.music.load(sounds["intro"])
pygame.mixer.music.play()
cv2.waitKey(2000)  # Wait for intro sound to finish

prev_time = time.time()
win = False

def calculate_motion(landmarks1, landmarks2):
    """Calculate motion by summing up the distance between corresponding landmarks."""
    if landmarks1 is None or landmarks2 is None:
        return 0

    motion = sum(
        ((landmarks1[i].x - landmarks2[i].x) ** 2 + (landmarks1[i].y - landmarks2[i].y) ** 2) ** 0.5
        for i in range(len(landmarks1))
        if landmarks1[i].visibility > 0.5 and landmarks2[i].visibility > 0.5
    )
    return motion

try:
    while cap.isOpened() and TIMER >= 0:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for a natural mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        # Draw landmarks on the frame
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the game timer
        show_frame = cv2.resize(green if is_green_light else red, (0, 0), fx=0.5, fy=0.5)
        cv2.putText(show_frame, f'Timer: {TIMER}', (20, 50), font, 1, (0, 0, 255), 2)

        # Check if time to switch light
        cur_time = time.time()
        if cur_time - prev_time >= random.randint(5, 10):  # Random delay between 5 to 10 seconds
            prev_time = cur_time
            TIMER -= 1
            is_green_light = not is_green_light  # Switch light

            # Play corresponding light sound
            if is_green_light:
                pygame.mixer.music.load(sounds["green_light"])
            else:
                pygame.mixer.music.load(sounds["red_light"])
            pygame.mixer.music.play()

            # Store previous landmarks for motion detection
            prev_landmarks_p1 = result.pose_landmarks.landmark if result.pose_landmarks else None
            prev_landmarks_p2 = prev_landmarks_p1  # Simulate second player (for webcam simplification)

        # Detect movement during red light
        if not is_green_light and result.pose_landmarks:
            motion_p1 = calculate_motion(prev_landmarks_p1, result.pose_landmarks.landmark)
            motion_p2 = calculate_motion(prev_landmarks_p2, result.pose_landmarks.landmark)
            print(f"Detected motion - Player 1: {motion_p1}, Player 2: {motion_p2}")  # For debugging

            # Check motion thresholds for both players
            if motion_p1 > motion_threshold:
                player_lost[0] = True  # Player 1 lost
            if motion_p2 > motion_threshold:
                player_lost[1] = True  # Player 2 lost

            # If any player loses, show the kill frame and play sound
            if any(player_lost):
                pygame.mixer.music.load(sounds["kill"])
                pygame.mixer.music.play()
                for _ in range(60):  # Display kill frame for ~3 seconds
                    cv2.imshow("Squid Game", cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
                    cv2.waitKey(50)
                break  # End the game if any player loses

        # Display the current frame and webcam view
        cam_show = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        cam_h, cam_w = cam_show.shape[:2]
        show_frame[0:cam_h, -cam_w:] = cam_show  # Place webcam in the top right corner
        cv2.imshow("Squid Game", show_frame)

        # Win condition
        if TIMER == 0 and is_green_light:
            win = True
            break

        # Handle quit event
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # After game logic: Win or lose scenario
    if win:
        pygame.mixer.music.load(sounds["win"])
        pygame.mixer.music.play()
        for _ in range(60):  # Display winner frame for 3 seconds
            cv2.imshow("Squid Game", cv2.resize(winner, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(50)
    elif any(player_lost):
        print("Player eliminated!")  # For debugging purposes
    else:
        # If the player runs out of time but didn't move during red light
        pygame.mixer.music.load(sounds["kill"])
        pygame.mixer.music.play()
        for _ in range(60):  # Display kill frame for 3 seconds
            cv2.imshow("Squid Game", cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(50)

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
