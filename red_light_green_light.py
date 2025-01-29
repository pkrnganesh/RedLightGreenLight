import cv2
import time
import pygame
from random import randint
import pyttsx3
import mediapipe as mp  # For pose detection
import os

# Initialize Pygame for playing music
pygame.init()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# File paths and settings
GAME_SONG = "sounds/game_song.mp3"
SNAPSHOT_FOLDER = "snapshots"
motion_threshold = 0.30  # Threshold for landmark movement

# Ensure snapshot directory exists
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Players and motion tracking
players = ["Player1", "Player2", "Player3"]
eliminated = []
prev_landmarks = {player: None for player in players}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize webcam
cap = cv2.VideoCapture(1)
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

# Function to wait for user to press space to continue
def wait_for_space():
    while True:
        cv2.putText(frame, "Press 'Space' to continue", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Game", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):  # Space key to continue
            break
        elif key == ord('q'):  # Quit the game
            raise KeyboardInterrupt

# Game loop
try:
    print("Game Starting! Press 'Space' to start.")

    pygame.mixer.music.load(GAME_SONG)
    
    while players:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        wait_for_space()

        count = 0
        play_duration = randint(1, 2)  # Random duration for the song to play

        pygame.mixer.music.play()
        frame_count = 4 * 24 * play_duration  # Frame count calculation based on duration

        while count < frame_count:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            count += 1
            frame_height, frame_width, _ = frame.shape
            segment_width = frame_width // len(players)  # Dynamic segmentation based on player count

            # Process each player segment separately
            for i, player in enumerate(players):
                start_x = i * segment_width
                end_x = start_x + segment_width
                player_frame = frame[:, start_x:end_x]

                rgb_part = cv2.cvtColor(player_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_part)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(player_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    prev_landmarks[player] = landmarks

                # Merge processed parts back to original frame
                frame[:, start_x:end_x] = player_frame

            # Draw segment lines
            for i in range(1, len(players)):
                cv2.line(frame, (i * segment_width, 0), (i * segment_width, frame_height), (0, 255, 0), 2)

            cv2.imshow("Free to Move", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        pygame.mixer.music.stop()
        print("Song stopped! Detecting motion...")

        count1 = 0
        players_to_remove = []
        while count1 < 5 * 24:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            count1 += 1
            frame_height, frame_width, _ = frame.shape
            segment_width = frame_width // len(players)

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

                # Merge processed parts back to original frame
                frame[:, start_x:end_x] = player_frame

            # Remove eliminated players after processing
            for player in players_to_remove:
                players.remove(player)

            # Draw segment lines
            for i in range(1, len(players)):
                cv2.line(frame, (i * segment_width, 0), (i * segment_width, frame_height), (0, 255, 0), 2)

            # Display elimination messages
            for idx, player in enumerate(eliminated):
                cv2.putText(frame, f"{player} Eliminated", (50, 50 + 30 * idx),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Detecting", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    print("Game Over! All players eliminated.")

except KeyboardInterrupt:
    print("Game interrupted manually!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


import cv2
import time
import pygame
from random import randint
import pyttsx3
import mediapipe as mp  # For pose detection
import os

# Initialize Pygame for playing music
pygame.init()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# File paths and settings
GAME_SONG = "sounds/game_song.mp3"
SNAPSHOT_FOLDER = "snapshots"
motion_threshold = 0.30  # Threshold for landmark movement

# Ensure snapshot directory exists
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Players and motion tracking
players = ["Player1"]
eliminated = []
prev_landmarks = {player: None for player in players}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
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

# Function to wait for user to press space to continue
def wait_for_space():
    while True:
        cv2.putText(frame, "Press 'Space' to continue", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Game", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):  # Space key to continue
            break
        elif key == ord('q'):  # Quit the game
            raise KeyboardInterrupt

# Game loop
try:
    print("Game Starting! Press 'Space' to start.")

    pygame.mixer.music.load(GAME_SONG)

    while players:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        wait_for_space()

        play_duration = randint(1, 2)  # Random duration for the song to play
        pygame.mixer.music.play()
        frame_count = 4 * 24 * play_duration  # Frame count calculation based on duration

        for count in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                prev_landmarks[players[0]] = landmarks

            cv2.imshow("Free to Move", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        pygame.mixer.music.stop()
        print("Song stopped! Detecting motion...")

        for count1 in range(5 * 24):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Detect motion for the player
                motion = detect_landmark_motion(prev_landmarks.get(players[0]), landmarks)
                print(f"{players[0]} Motion: {motion}")

                if motion > motion_threshold and players[0] in players:
                    eliminated.append(players[0])
                    print(f"Eliminated: {players[0]}")
                    read_out_names([players[0]])
                    save_snapshot(players[0], frame)
                    players.pop(0)  # Remove the player from the list

                # Display elimination message
                for idx, player in enumerate(eliminated):
                    cv2.putText(frame, f"{player} Eliminated", (50, 50 + 30 * idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Detecting", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    print("Game Over! You were eliminated.")

except KeyboardInterrupt:
    print("Game interrupted manually!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()