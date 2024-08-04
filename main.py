from algorithms import get_random_video, mog1, mog2, knn
from annotator import VideoAnnotator

INPUT_FOLDER = "./input/dataset1"
OUTPUT_FOLDER = "./output/"
OUTPUT_JSON = "output.json"
HEADLESS = False

LEARNING_RATE = -1      # alpha
HISTORY = 200           # t
N_MIXTURES = 5          # K (number of gaussians)
BACKGROUND_RATIO = 0.1  # Gaussian threshold
NOISE_SIGMA = 1
MOTION_ENERGY_THRESHOLD = 0.01

def banner():
    print("""
  ___ _              __  __     _   _          ___      _          _   _          
 / __| |___ ___ _ __|  \/  |___| |_(_)___ _ _ |   \ ___| |_ ___ __| |_(_)___ _ _  
 \__ \ / -_) -_) '_ \ |\/| / _ \  _| / _ \ ' \| |) / -_)  _/ -_) _|  _| / _ \ ' \ 
 |___/_\___\___| .__/_|  |_\___/\__|_\___/_||_|___/\___|\__\___\__|\__|_\___/_||_|
               |_|                                                                
                                     
Version: 1.0
Author:  Giacomo Tezza
Github:  https://github.com/GiacomoTezza

""")

def annotateMog1(name, annotator, cap, learning_rate, history, n_mixtures, background_ratio, noise_sigma, motion_energy_threshold):
    print(f"[MOG1][{name}] Running...")
    mog1_data = mog1(cap, learning_rate, history, n_mixtures, background_ratio, noise_sigma, motion_energy_threshold, headless=HEADLESS)
    annotator.add_motion_data(
        algorithm_name=f"MOG1-{name}",
        parameters={
            "LearningRate": learning_rate,
            "History": history,
            "NumberMixtures": n_mixtures,
            "BackgroundRatio": background_ratio,
            "NoiseSigma": noise_sigma,
            "MotionEnergyThreshold": motion_energy_threshold
        },
        motion_data=mog1_data
    )
    print(f"[MOG1][{name}] Completed!\n")


def annotateMog2(name, annotator, cap, learning_rate, motion_energy_threshold):
    print(f"[MOG2][{name}] Running...")
    mog2_data = mog2(cap, learning_rate, motion_energy_threshold, headless=HEADLESS)
    annotator.add_motion_data(
        algorithm_name=f"MOG2-{name}",
        parameters={
            "LearningRate": learning_rate,
            "MotionEnergyThreshold": motion_energy_threshold
        },
        motion_data=mog2_data
    )
    print(f"[MOG2][{name}] Completed!\n")


def annotateKnn(name, annotator, cap, learning_rate, dist2_threshold, detect_shadows, motion_energy_threshold):
    print(f"[KNN][{name}] Running...")
    knn_data = knn(cap, learning_rate, dist2_threshold, detect_shadows, motion_energy_threshold, headless=HEADLESS)
    annotator.add_motion_data(
        algorithm_name=f"KNN-{name}",
        parameters={
            "LearningRate": learning_rate,
            "DistanceToThreshold": dist2_threshold,
            "DetectShadows": detect_shadows,
            "MotionEnergyThreshold": motion_energy_threshold
        },
        motion_data=knn_data
    )
    print(f"[KNN][{name}] Completed!\n")


def main():
    banner()
    cap, filename = get_random_video(INPUT_FOLDER)
    annotator = VideoAnnotator(filename)
    annotateMog1("param1", annotator, cap, LEARNING_RATE, HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA, MOTION_ENERGY_THRESHOLD)
    annotateMog2("param1", annotator, cap, LEARNING_RATE, MOTION_ENERGY_THRESHOLD)
    annotateKnn("param1", annotator, cap, LEARNING_RATE, 400, False, MOTION_ENERGY_THRESHOLD)
    cap.release()
    annotator.save_to_json(OUTPUT_FOLDER+OUTPUT_JSON)

if __name__ == "__main__":
    main()
