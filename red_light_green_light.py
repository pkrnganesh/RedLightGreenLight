import cv2
import time
import pygame
from random import randint
import pyttsx3
import mediapipe as mp
import os
from PIL import Image, ImageSequence
import numpy as np

# Initialize Pygame for playing music
pygame.init()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# File paths and settings
GAME_SONG = "sounds/game_song.mp3"
SNAPSHOT_FOLDER = "snapshots"
GIF_FILE = "doll_gif_frames/giphy.gif"
motion_threshold = 0.30  # Threshold for landmark movement

# Ensure snapshot directory exists
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Load GIF frames
gif_frames = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in ImageSequence.Iterator(Image.open(GIF_FILE))]

# Players and motion tracking
players = ["Player1", "Player2"]
eliminated = []
prev_landmarks = {player: None for player in players}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Helper function to detect motion
def detect_landmark_motion(landmarks_prev, landmarks_curr):
    if landmarks_prev is None:
        return 0  # If no previous landmarks exist
    total_motion = sum(
        ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5
        for prev, curr in zip(landmarks_prev, landmarks_curr)
        if prev.visibility > 0.5 and curr.visibility > 0.5
    )
    return total_motion

# Helper function to save a snapshot
def save_snapshot(player, frame):
    filepath = os.path.join(SNAPSHOT_FOLDER, f"{player}_snapshot.jpg")
    cv2.imwrite(filepath, frame)
    print(f"Snapshot saved for {player}: {filepath}")

# Helper function to read out names
def read_out_names(names):
    for name in names:
        engine.say(name + " was eliminated.")
    engine.runAndWait()

# Function to display GIF and control its state
def display_gif(frame_index, reverse=False):
    frame_index = frame_index % len(gif_frames)
    if reverse:
        frame_index = len(gif_frames) - frame_index - 1
    return gif_frames[frame_index]

# Function to resize and overlay the small webcam feed
def overlay_webcam_feed(gif_frame, webcam_frame):
    resized_webcam = cv2.resize(webcam_frame, (200, 150))  # Resize webcam feed to fit in bottom-left corner
    gif_frame[gif_frame.shape[0]-150:gif_frame.shape[0], 0:200] = resized_webcam  # Overlay in bottom-left corner
    return gif_frame

# Function to wait for user to press space to continue
def wait_for_space(frame):
    while True:
        cv2.putText(frame, "Press 'Space' to start", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Game", frame)
        key = cv2.waitKey(1) & 0xFF  # Check key every frame
        if key == ord(' '):  # Space key to continue
            break
        elif key == ord('q'):  # Quit the game
            raise KeyboardInterrupt

# Game loop
try:
    print("Game Starting! Press 'Space' to start.")
    pygame.mixer.music.load(GAME_SONG)

    frame_index = 0
    while players:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Show the initial frame and wait for user input
        wait_for_space(frame)

        play_duration = randint(1, 2)  # Random duration for the song to play

        pygame.mixer.music.play()
        frame_count = 4 * 24 * play_duration  # Frame count calculation based on duration
        count = 0

        while count < frame_count:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            count += 1

            # Display the GIF's current frame
            gif_frame = display_gif(frame_index, reverse=False)
            gif_frame = cv2.resize(gif_frame, (800, 600))  # Resize the GIF to fullscreen (modify size as needed)

            # Overlay the small webcam feed in the bottom-left corner
            combined_frame = overlay_webcam_feed(gif_frame, frame)

            cv2.imshow("Free to Move", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

            frame_index += 1  # Move to the next GIF frame

        pygame.mixer.music.stop()
        print("Song stopped! Detecting motion...")

        players_to_remove = []
        detection_duration = 5 * 24  # Time to detect motion after the song stops
        count = 0

        while count < detection_duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            count += 1

            # Display the GIF frame in reverse (simulating the character looking back)
            gif_frame = display_gif(frame_index, reverse=True)
            gif_frame = cv2.resize(gif_frame, (800, 600))

            # Overlay the small webcam feed
            combined_frame = overlay_webcam_feed(gif_frame, frame)

            cv2.imshow("Detecting Motion", combined_frame)

            # Process player motion detection
            frame_height, frame_width, _ = frame.shape
            segment_width = frame_width // 2  # Divide frame into 2 regions for Player1 and Player2

            for i, player in enumerate(players):
                start_x = i * segment_width
                end_x = start_x + segment_width
                player_frame = frame[:, start_x:end_x]

                rgb_part = cv2.cvtColor(player_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_part)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(player_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Detect motion
                    motion = detect_landmark_motion(prev_landmarks.get(player), landmarks)
                    print(f"{player} Motion: {motion}")

                    if motion > motion_threshold:
                        players_to_remove.append(player)
                        eliminated.append(player)
                        print(f"Eliminated: {player}")
                        read_out_names([player])
                        save_snapshot(player, frame)

                    prev_landmarks[player] = landmarks

            # Remove eliminated players
            for player in players_to_remove:
                players.remove(player)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    print("Game Over! All players eliminated.")

except KeyboardInterrupt:
    print("Game interrupted manually!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
