from algorithms import *
from annotator import *

INPUT_FOLDER = "./input/dataset2"
INPUT_VIDEO = "./input/dataset2/3.mp4"
OUTPUT_FOLDER = "./output/"
OUTPUT_JSON = "output.json"
HEADLESS = True

LEARNING_RATE = -1      # alpha
HISTORY = 200           # t
N_MIXTURES = 5          # K (number of gaussians)
BACKGROUND_RATIO = 0.1  # Gaussian threshold
NOISE_SIGMA = 1
MOTION_ENERGY_THRESHOLD = 0.01
HYSTERESIS = 25

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

def main():
    banner()
    # cap, filename = get_random_video(INPUT_FOLDER)
    cap, filename = get_video(INPUT_VIDEO)
    annotator = VideoAnnotator(filename, OUTPUT_FOLDER)
    # annotateMog1("param1", annotator, cap, 0.75, HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA, MOTION_ENERGY_THRESHOLD, HYSTERESIS, HEADLESS)
    # annotateMog1("param2", annotator, cap, 0.9, HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA, MOTION_ENERGY_THRESHOLD, HYSTERESIS, HEADLESS)
    # annotateMog1("param3", annotator, cap, 1, HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA, MOTION_ENERGY_THRESHOLD, HYSTERESIS, HEADLESS)

    # annotateMog2("param1", annotator, cap, LEARNING_RATE, MOTION_ENERGY_THRESHOLD, HYSTERESIS, HEADLESS)
    # annotateKnn("param1", annotator, cap, LEARNING_RATE, 400, False, MOTION_ENERGY_THRESHOLD, HYSTERESIS, HEADLESS)

    grid_search_knn(annotator, filename)
    grid_search_mog2(annotator, filename)
    grid_search_mog1(annotator, filename)
    cap.release()
    annotator.save_to_json(OUTPUT_JSON)

if __name__ == "__main__":
    main()
