import cv2
import random
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from os import listdir, path
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

def get_random_video(directory):
    file = random.choice(listdir(directory))
    filename = f"{directory}/{path.normpath(file)}"
    print(f"[INPUT] {filename}\n")
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


def initialize_plot(max_frames, motion_energy_threshold):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, max_frames)
    ax.set_ylim(0, 1)
    motion_energy_line, = ax.plot([], [], label='Motion Energy')
    threshold_line, = ax.plot([0, max_frames], [motion_energy_threshold, motion_energy_threshold], 'r--', label='Threshold')
    motion_presence_line, = ax.plot([], [], label='Motion Presence')
    ax.legend()
    return fig, ax, motion_energy_line, threshold_line, motion_presence_line


def update_plot(motion_energy_line, motion_presence_line, motion_data):
    motion_energy_list = [data['motion_energy'] for data in motion_data]
    motion_presence_wave = np.array([1 if data['motion_presence'] else 0 for data in motion_data])
    motion_energy_line.set_xdata(np.arange(len(motion_energy_list)))
    motion_energy_line.set_ydata(motion_energy_list)
    motion_presence_line.set_xdata(np.arange(len(motion_presence_wave)))
    motion_presence_line.set_ydata(motion_presence_wave ** 2)
    plt.draw()
    plt.pause(0.001)