import cv2
import random
from screeninfo import get_monitors

def get_screen_resolution():
    monitors = get_monitors()
    if monitors:
        # Assuming the first monitor is the main display
        main_monitor = monitors[0]
        width = main_monitor.width
        height = main_monitor.height
        return width, height
    else:
        return None

def get_random_video():
    random_dir = str(random.randint(1, 28)).zfill(2)
    filename = f"../dataset/{random_dir}/rgb.avi"
    cap = cv2.VideoCapture(filename)
    return (cap, filename)

def calculate_window_positions(num_windows):
    # Get screen dimensions
    screen_width, screen_height = get_screen_resolution()

    if num_windows == 1:
        # Fullscreen window
        return {'fullscreen': (0, 0, screen_width, screen_height)}
    elif num_windows == 2:
        # Two windows side by side at the center
        window_width = screen_width // 2
        window_height = screen_height
        return {
            'left': (0, 0, window_width, window_height),
            'right': (window_width, 0, window_width, window_height)
        }
    elif num_windows == 3:
        # Three windows: top left, top right, bottom center
        window_width = screen_width // 2
        window_height = screen_height // 2
        return {
            'top_left': (0, 0, window_width, window_height),
            'top_right': (window_width, 0, window_width, window_height),
            'bottom_center': (screen_width // 4, screen_height // 2, window_width, window_height)
        }
    elif num_windows == 4:
        # Four windows: one for each corner
        window_width = screen_width // 2
        window_height = screen_height // 2
        return {
            'top_left': (0, 0, window_width, window_height),
            'top_right': (screen_width // 2, 0, window_width, window_height),
            'bottom_left': (0, screen_height // 2, window_width, window_height),
            'bottom_right': (screen_width // 2, screen_height // 2, window_width, window_height)
        }
    else:
        # Unsupported number of windows
        raise ValueError("Unsupported number of windows")
