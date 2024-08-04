import cv2
import numpy as np
from .utils import calculate_window_positions
from tqdm import trange

def knn(cap, learning_rate, dist2_threshold=400.0, detect_shadows=True, motion_energy_threshold=0.01, headless=True):
    fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=dist2_threshold, detectShadows=detect_shadows)
    motion_data = []

    if not headless:
        window_positions_knn = calculate_window_positions(2)
        for window_name, (x, y, width, height) in window_positions_knn.items():
            cv2.namedWindow(window_name)
            cv2.resizeWindow(window_name, width, height)
            cv2.moveWindow(window_name, x, y)

    for i in trange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If video end reached
        if not ret:
            break

        fgmask = fgbg.apply(frame, learning_rate)

        # Calculate motion degree as the proportion of non-zero pixels in the foreground mask
        motion_energy = np.count_nonzero(fgmask) / fgmask.size
        motion_presence = motion_energy > motion_energy_threshold

        motion_data.append({
            'frame': i,
            'motion_presence': motion_presence,
            'motion_energy': motion_energy
        })

        if not headless:
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
    
    return motion_data
