from algorithms import get_random_video, mog1, mog2
from annotator import VideoAnnotator

MAX_FRAMES = 2000
LEARNING_RATE = -1      # alpha
HISTORY = 200           # t
N_MIXTURES = 5          # K (number of gaussians)
BACKGROUND_RATIO = 0.1  # Gaussian threshold
NOISE_SIGMA = 1
MOTION_ENERGY_THRESHOLD = 0.01


def annotateMog1(name, annotator, cap, max_frames, learning_rate, history, n_mixtures, background_ratio, noise_sigma, motion_energy_threshold):
    mog1_data = mog1(cap, max_frames, learning_rate, history, n_mixtures, background_ratio, noise_sigma, motion_energy_threshold, headless=True)
    annotator.add_motion_data(
        algorithm_name=f"MOG1-{name}",
        parameters={
            "MaxFrames": max_frames,
            "LearningRate": learning_rate,
            "History": history,
            "NumberMixtures": n_mixtures,
            "BackgroundRatio": background_ratio,
            "NoiseSigma": noise_sigma,
            "MotionEnergyThreshold": motion_energy_threshold
        },
        motion_data=mog1_data
    )


def annotateMog2(name, annotator, cap, max_frames, learning_rate, motion_energy_threshold):
    mog2_data = mog2(cap, max_frames, learning_rate, motion_energy_threshold, headless=True)
    annotator.add_motion_data(
        algorithm_name=f"MOG2-{name}",
        parameters={
            "MaxFrames": max_frames,
            "LearningRate": learning_rate,
            "MotionEnergyThreshold": motion_energy_threshold
        },
        motion_data=mog2_data
    )


def main():
    cap, filename = get_random_video()
    annotator = VideoAnnotator(filename)
    annotateMog1("param1", annotator, cap, MAX_FRAMES, LEARNING_RATE, HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA, MOTION_ENERGY_THRESHOLD)
    annotateMog2("param1", annotator, cap, MAX_FRAMES, LEARNING_RATE, MOTION_ENERGY_THRESHOLD)
    cap.release()
    annotator.save_to_json("./output.json")

if __name__ == "__main__":
    main()
