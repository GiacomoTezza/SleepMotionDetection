import cv2
from .utils import calculate_window_positions

def mog1(cap, max_frames, learning_rate, history, n_mixtures, background_ratio, noise_sigma):
    window_positions_mog1 = calculate_window_positions(2)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history, n_mixtures, background_ratio, noise_sigma)

    for window_name, (x, y, width, height) in window_positions_mog1.items():
        cv2.namedWindow(window_name)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, x, y)

    for i in range(max_frames):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If video end reached
        if not ret:
            break

        fgmask = fgbg.apply(frame, learning_rate)

        to_render = [fgmask, frame]
        for window_name, (x, y, w, h) in window_positions_mog1.items():
            cv2.imshow(window_name, to_render.pop())
            cv2.resizeWindow(window_name, w, h)

        cv2.imshow('left', fgmask)
        cv2.imshow('right', frame)

        # Wait and exit if q is pressed
        if cv2.waitKey(10) == ord('q') or not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def mog2(cap, max_frames, learning_rate):
    window_positions_mog2 = calculate_window_positions(3)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    for window_name, (x, y, width, height) in window_positions_mog2.items():
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, x, y)

    for i in range(max_frames):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If video end reached
        if not ret:
            break

        fgmask = fgbg.apply(frame, learning_rate)
        bg = fgbg.getBackgroundImage()

        to_render = [fgmask, bg, frame]
        for window_name, (x, y, w, h) in window_positions_mog2.items():
            cv2.imshow(window_name, to_render.pop())
            cv2.resizeWindow(window_name, w, h//2)

        # Wait and exit if q is pressed
        if cv2.waitKey(10) == ord('q') or not ret:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

