import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import calculate_window_positions, initialize_plot, update_plot
from tqdm import trange

def knn(cap, learning_rate, dist2_threshold=400.0, detect_shadows=True, motion_energy_threshold=0.01, hysteresis=200, headless=True, show_progress=True):
    fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=dist2_threshold, detectShadows=detect_shadows)
    motion_data = []
    frames_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hysteresis_counter = 0

    if not headless:
        window_positions_knn = calculate_window_positions(2)
        for window_name, (x, y, width, height) in window_positions_knn.items():
            cv2.namedWindow(window_name)
            cv2.resizeWindow(window_name, width, height)
            cv2.moveWindow(window_name, x, y)
        
        # Real-Time plot setup
        fig, ax, motion_energy_line, threshold_line, motion_presence_line = initialize_plot(frames_tot, motion_energy_threshold)

    if show_progress:
        frame_iter = trange(frames_tot, desc="Processing Frames")
    else:
        frame_iter = range(frames_tot)
    
    for i in frame_iter:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If video end reached
        if not ret:
            break

        fgmask = fgbg.apply(frame, learning_rate)

        # Apply Morphological Opening to reduce noise and improve contour detection
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # Median blur
        fgmask = cv2.medianBlur(fgmask, 5)

        # Calculate motion degree as the proportion of non-zero pixels in the foreground mask
        motion_energy = np.count_nonzero(fgmask) / fgmask.size
        # Hysteresis
        if motion_energy > motion_energy_threshold:
            hysteresis_counter = hysteresis  # Reset counter when motion is detected
        else:
            hysteresis_counter -= 1

        motion_presence = hysteresis_counter > 0

        motion_data.append({
            'frame': i,
            'motion_presence': motion_presence,
            'motion_energy': motion_energy
        })

        if not headless:
            # Real-Time plot
            update_plot(motion_energy_line, motion_presence_line, motion_data)

            to_render = [fgmask, frame]
            for window_name, (x, y, w, h) in window_positions_knn.items():
                cv2.imshow(window_name, to_render.pop())
                cv2.resizeWindow(window_name, w, h)

            cv2.imshow('left', fgmask)
            cv2.imshow('right', frame)

            # Wait and exit if q is pressed
            if cv2.waitKey(10) == ord('q') or not ret:
                break

    # When everything done, restart the capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not headless:
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close()
    
    return motion_data
