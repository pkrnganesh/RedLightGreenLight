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
    "intro": "sounds/squid.mp3",
    "kill": "sounds/kill.mp3",
    "win": "sounds/win.mp3",
    "green_light": "sounds/green.mp3",
    "red_light": "sounds/red.mp3",
    "eliminated": "sounds/eliminated.mp3",
    "winner_sound": "sounds/winner.mp3" # Add new sound for winning
}

# Load frames for the game visuals
mylist = os.listdir(folderPath)
graphic = [cv2.imread(f'{folderPath}/{imPath}') for imPath in sorted(mylist)]
green = graphic[5]
red = graphic[6]
kill = graphic[2]  # Blood/elimination frame
winner = graphic[3]
intro = graphic[4]

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Game timer and movement detection setup
TIMER_MAX = 20  # Increased to 20 seconds
motion_threshold = 0.30  # Motion threshold for a single player
font = cv2.FONT_HERSHEY_SIMPLEX
is_green_light = True
prev_landmarks = None  # To store previous pose landmarks for the player
player_lost = False  # Track whether the player lost

# Fullscreen setup
cv2.namedWindow("Squid Game", cv2.WINDOW_NORMAL)  # Allow window resizing
cv2.setWindowProperty("Squid Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set to fullscreen
screen_width, screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h  # Get screen dimensions

# Start button setup - Left bottom corner
start_button_color = (50, 205, 50)  # LimeGreen
start_button_hover_color = (0, 128, 0) # DarkGreen
start_button_text_color = (255, 255, 255)  # White
start_button_width = 300
start_button_height = 75
start_button_x = screen_width - start_button_width - 350  # Move to right
start_button_y = screen_height - start_button_height - 100  # Keep it at the bottom
  # Position from bottom edge
start_button_font = cv2.FONT_HERSHEY_DUPLEX  # More readable font
start_button_text = "Start Game"
start_button_is_hovered = False

# Eliminate text setup
eliminate_text_color = (255, 0, 0)  # Red
eliminate_text_font = cv2.FONT_HERSHEY_DUPLEX
eliminate_text_size = 2 # Enlarge
eliminate_text_thickness = 3 #Enlarge
eliminate_text = "You are Eliminated!"
eliminate_text_duration = 3  # Seconds

# Initialize the cursor as arrow
pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW)

# Enter button setup - Top right corner
enter_button_color = (50, 205, 50)# CornflowerBlue
enter_button_hover_color = (70, 130, 180)  # SteelBlue
enter_button_text_color = (255, 255, 255)  # White
enter_button_width = 150
enter_button_height = 50
enter_button_x = screen_width - enter_button_width - 20
enter_button_y = screen_height - enter_button_height - 20  # Position from bottom edge
enter_button_font = cv2.FONT_HERSHEY_DUPLEX
enter_button_text = "Enter"
enter_button_is_hovered = False

#Flag to determine if enter is avaiable
show_enter_button = False

def draw_start_button(frame):
    button_color = start_button_hover_color if start_button_is_hovered else start_button_color
    cv2.rectangle(frame, (start_button_x, start_button_y), (start_button_x + start_button_width, start_button_y + start_button_height), button_color, -1)
    text_size = cv2.getTextSize(start_button_text, start_button_font, 1.5, 3)[0]  #Increased Size
    text_x = start_button_x + (start_button_width - text_size[0]) // 2
    text_y = start_button_y + (start_button_height + text_size[1]) // 2
    cv2.putText(frame, start_button_text, (text_x, text_y), start_button_font, 1.5, start_button_text_color, 3)  #Increased Size

def is_start_button_clicked(x, y):
    return start_button_x <= x <= start_button_x + start_button_width and start_button_y <= y <= start_button_y + start_button_height

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

# Initialize variables outside the loop
game_running = False
cap = None  # Define cap here
win = False
prev_time = time.time()
TIMER = TIMER_MAX  # Initialize the timer here
winner_displayed = False # To track if the winner frame is displayed

def reset_game_state():
    """Resets all game-related variables to their initial states."""
    global game_running, win, prev_time, TIMER, is_green_light, player_lost, prev_landmarks, show_enter_button, winner_displayed
    game_running = False
    win = False
    prev_time = time.time()
    TIMER = TIMER_MAX
    is_green_light = True
    player_lost = False
    prev_landmarks = None
    show_enter_button = False
    winner_displayed = False
    # Reset cursor to arrow
    pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW)

def draw_enter_button(frame):
    button_color = enter_button_hover_color if enter_button_is_hovered else enter_button_color
    cv2.rectangle(frame, (enter_button_x, enter_button_y), (enter_button_x + enter_button_width, enter_button_y + enter_button_height), button_color, -1)
    text_size = cv2.getTextSize(enter_button_text, enter_button_font, 1, 2)[0]
    text_x = enter_button_x + (enter_button_width - text_size[0]) // 2
    text_y = enter_button_y + (enter_button_height + text_size[1]) // 2
    cv2.putText(frame, enter_button_text, (text_x, text_y), enter_button_font, 1, enter_button_text_color, 2)

def is_enter_button_clicked(x, y):
    return enter_button_x <= x <= enter_button_x + enter_button_width and enter_button_y <= y <= enter_button_y + enter_button_height

def mouse_callback(event, x, y, flags, param):
    global game_running, TIMER, player_lost, win, prev_time, cap, is_green_light, start_button_is_hovered, enter_button_is_hovered, show_enter_button, winner_displayed

    if event == cv2.EVENT_MOUSEMOVE:
        # Check if mouse is over start button
        if not game_running and is_start_button_clicked(x, y):
            if not start_button_is_hovered:
                start_button_is_hovered = True
                pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_HAND)
        elif not game_running:
            if start_button_is_hovered:
                start_button_is_hovered = False
                pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW)

        #Check if mouse is over enter button
        if game_running and show_enter_button and is_enter_button_clicked(x, y):
            if not enter_button_is_hovered:
                enter_button_is_hovered = True
                pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_HAND)
        elif game_running and show_enter_button:
            if enter_button_is_hovered:
                enter_button_is_hovered = False
                pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if not game_running and is_start_button_clicked(x, y):
            print("Start button clicked!")
            game_running = True
            TIMER = TIMER_MAX
            player_lost = False
            win = False
            prev_time = time.time()
            is_green_light = True
            show_enter_button = True #Show Enter Button
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Error: Could not open webcam")
                    game_running = False
                    return
            pygame.mixer.music.load(sounds["intro"])
            pygame.mixer.music.play()
            cv2.waitKey(2000)
        elif game_running and show_enter_button and is_enter_button_clicked(x, y):
            #Player Won
            win = True
            game_running = False # Exit game loop


while True:
    if not game_running:
        # Display start screen
        frame = intro.copy() # Use a copy to avoid modifying the original intro image
        frame = cv2.resize(frame, (screen_width, screen_height)) #resize the image
        draw_start_button(frame)
        cv2.imshow("Squid Game", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.setMouseCallback("Squid Game", mouse_callback)

    else:
        # Game logic (inside try-finally block)
        try:
            if cap is None or not cap.isOpened():
                print("Error: Webcam not accessible")
                game_running = False # Exit game loop
                reset_game_state()  # Reset all game variables
                continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                game_running = False
                reset_game_state()  # Reset all game variables
                continue

            frame = cv2.flip(frame, 1)  # Flip horizontally for a natural mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            # Draw landmarks on the frame
            if result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the game timer
            show_frame = cv2.resize(green if is_green_light else red, (screen_width, screen_height)) #resize image
            cv2.putText(show_frame, f'Timer: {TIMER}', (20, 70), font, 3, (0, 0, 255), 4)

            #Check if enter button have to be displayed
            if show_enter_button:
                draw_enter_button(show_frame) #draw enter button

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
                if result.pose_landmarks:
                    prev_landmarks = result.pose_landmarks.landmark

            # Detect movement during red light
            if not is_green_light and result.pose_landmarks:
                motion = calculate_motion(prev_landmarks, result.pose_landmarks.landmark)
                print(f"Detected motion - Player: {motion}")  # For debugging

                # Check motion threshold for the player
                if motion > motion_threshold:
                    player_lost = True  # Player lost

                # If any player loses, show the kill frame and play sound
                if player_lost:
                    pygame.mixer.music.load(sounds["kill"])
                    pygame.mixer.music.play()
                    kill_frame = cv2.resize(kill, (screen_width, screen_height)) #resize image
                    text_size = cv2.getTextSize(eliminate_text, eliminate_text_font, eliminate_text_size, eliminate_text_thickness)[0]
                    text_x = (screen_width - text_size[0]) // 2
                    text_y = screen_height // 2
                    cv2.putText(kill_frame, eliminate_text, (text_x, text_y), eliminate_text_font, eliminate_text_size, eliminate_text_color, eliminate_text_thickness)

                    cv2.imshow("Squid Game", kill_frame)  # Show the kill frame immediately
                    cv2.waitKey(1)
                    time.sleep(0.5)  # Short pause to allow the kill sound to play

                    pygame.mixer.music.load(sounds["eliminated"])
                    pygame.mixer.music.play()  # Play the eliminated sound


                    start_time = time.time()
                    while time.time() - start_time < eliminate_text_duration:
                        cv2.imshow("Squid Game", kill_frame)
                        cv2.waitKey(1) # Necessary for the window to update

                    game_running = False # Return to start screen
                    reset_game_state()  # Reset all game variables
                    continue  # Skip to the next iteration of the main loop

            # Display the current frame and webcam view
            cam_show = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            cam_h, cam_w = cam_show.shape[:2]
            show_frame[0:cam_h, -cam_w:] = cam_show  # Place webcam in the top right corner
            cv2.imshow("Squid Game", show_frame)

            # Win condition
            if TIMER == 0 and is_green_light:
                win = True
                game_running = False
                continue

            # Handle quit event
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        finally: # This part runs after either winning, losing or pressing 'q'
            if not game_running:  # Only run if game is over
                if win:
                    pygame.mixer.music.load(sounds["winner_sound"]) #play u are winner sound
                    pygame.mixer.music.play()
                    winner_frame = cv2.resize(winner, (screen_width, screen_height))
                    start_time = time.time()
                    while time.time() - start_time < 3:  # 3 second display
                        cv2.imshow("Squid Game", winner_frame)
                        cv2.waitKey(1)
                    reset_game_state()  # Reset after showing winner frame

                elif player_lost:
                    print("Player eliminated!")  # For debugging purposes

                else: # ran out of time but didn't move
                    pygame.mixer.music.load(sounds["eliminated"])
                    pygame.mixer.music.play()
                    kill_frame = cv2.resize(kill, (screen_width, screen_height)) #resize image
                    start_time = time.time()
                    while time.time() - start_time < eliminate_text_duration:
                        cv2.imshow("Squid Game", kill_frame)
                        cv2.waitKey(1)
                    reset_game_state() #reset game

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    if key == ord('q'):
        break

# Outside the while loop to ensure resources are released only once.
if cap is not None and cap.isOpened():
    cap.release()

cv2.destroyAllWindows()
pygame.quit()