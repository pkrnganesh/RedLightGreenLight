import cv2
import time
import pygame
from random import randint
import pyttsx3
import mediapipe as mp
import os
from itertools import cycle

# Initialize Pygame for playing music
pygame.init()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# File paths and settings
GAME_SONG = "sounds/game_song.mp3"
STATIC_DOLL_IMAGE = "doll_gif_frames/static_doll_frame.png"  # Static image for the doll facing forward
SNAPSHOT_FOLDER = "snapshots"
motion_threshold = 0.30  # Threshold for landmark movement

# Ensure snapshot directory exists
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Single player tracking
players = ["Player1"]
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

# Load GIF frames and static image
gif_frames = [cv2.imread(f"doll_gif_frames/frame_{i}.png") for i in range(1, 20)]  # Load GIF frames
frame_cycle = cycle(gif_frames)  # Cycle through GIF frames
static_doll_frame = cv2.imread(STATIC_DOLL_IMAGE)  # Load the static image

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

# Game loop
try:
    print("Game Starting! Press 'Space' to start.")
    pygame.mixer.music.load(GAME_SONG)

    while players:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Resize the static doll image to match half the screen
        height, width, _ = frame.shape
        half_width = width // 2
        static_doll_resized = cv2.resize(static_doll_frame, (half_width, height))

        # Determine the song duration (random between 3 to 6 seconds)
        song_duration = randint(3, 6)
        pygame.mixer.music.play(loops=0)  # Play the song once

        # Play the song while displaying the static doll image
        start_time = time.time()
        while time.time() - start_time < song_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize the player frame to match the right half of the screen
            player_frame_resized = cv2.resize(frame, (half_width, height))

            # Concatenate the static doll and player frames
            combined_frame = cv2.hconcat([static_doll_resized, player_frame_resized])

            # Display the message indicating the doll is facing forward
            cv2.putText(combined_frame, "Doll is facing forward! Move freely.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the combined view
            cv2.imshow("Game - Doll and Player", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        pygame.mixer.music.stop()  # Stop the music when the doll turns back
        print("Doll turned back! Detecting motion...")

        # Play the GIF frames to show the doll turning back
        for gif_frame in gif_frames:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize the player frame to match the right half of the screen
            player_frame_resized = cv2.resize(frame, (half_width, height))

            # Resize the GIF frame
            gif_resized = cv2.resize(gif_frame, (half_width, height))

            # Concatenate the GIF frame and player detection frame
            combined_frame = cv2.hconcat([gif_resized, player_frame_resized])

            # Display the message indicating the doll is turned back
            cv2.putText(combined_frame, "Doll turned back! Freeze!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Game - Doll and Player", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        # Detect motion for 4 seconds during GIF playback
        detection_start_time = time.time()
        while time.time() - detection_start_time < 4:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize the player frame to match the right half of the screen
            player_frame_resized = cv2.resize(frame, (half_width, height))

            # Concatenate the last GIF frame and player detection frame
            combined_frame = cv2.hconcat([gif_resized, player_frame_resized])

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Detect motion
                motion = detect_landmark_motion(prev_landmarks[players[0]], landmarks)
                prev_landmarks[players[0]] = landmarks  # Update previous landmarks

                print(f"{players[0]} Motion: {motion}")

                if motion > motion_threshold and players[0] in players:
                    eliminated.append(players[0])
                    print(f"Eliminated: {players[0]}")
                    read_out_names([players[0]])
                    save_snapshot(players[0], frame)
                    players.remove(players[0])
                    break  # Re-evaluate players list

            # Show the combined frame
            cv2.imshow("Game - Doll and Player", combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

    print("Game Over! Player eliminated.")

except KeyboardInterrupt:
    print("Game interrupted manually!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
