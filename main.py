from algorithms import get_random_video, mog1, mog2

MAX_FRAMES = 1000
LEARNING_RATE = -1  # alpha
HISTORY = 200       # t
N_MIXTURES = 5      # K (number of gaussians)
BACKGROUND_RATIO = 0.1 # Gaussian threshold
NOISE_SIGMA = 1 

def main():
    cap = get_random_video()
    # mog1(cap, MAX_FRAMES, LEARNING_RATE, HISTORY, N_MIXTURES, BACKGROUND_RATIO, NOISE_SIGMA)
    mog2(cap, MAX_FRAMES, LEARNING_RATE)

if __name__ == "__main__":
    main()
