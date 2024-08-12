from algorithms import mog1, mog2, knn

def annotateMog1(name, annotator, cap, learning_rate, history, n_mixtures, background_ratio, noise_sigma, motion_energy_threshold, hysteresis, headless):
    print(f"[MOG1][{name}] Running...")
    mog1_data = mog1(cap, learning_rate, history, n_mixtures, background_ratio, noise_sigma, motion_energy_threshold, hysteresis, headless)
    annotator.add_motion_data(
        algorithm_name=f"MOG1-{name}",
        parameters={
            "LearningRate": learning_rate,
            "History": history,
            "NumberMixtures": n_mixtures,
            "BackgroundRatio": background_ratio,
            "NoiseSigma": noise_sigma,
            "MotionEnergyThreshold": motion_energy_threshold,
            "Hysteresis": hysteresis
        },
        motion_data=mog1_data
    )
    print(f"[MOG1][{name}] Completed!\n")


def annotateMog2(name, annotator, cap, learning_rate, motion_energy_threshold, hysteresis, headless):
    print(f"[MOG2][{name}] Running...")
    mog2_data = mog2(cap, learning_rate, motion_energy_threshold, hysteresis, headless)
    annotator.add_motion_data(
        algorithm_name=f"MOG2-{name}",
        parameters={
            "LearningRate": learning_rate,
            "MotionEnergyThreshold": motion_energy_threshold,
            "Hysteresis": hysteresis
        },
        motion_data=mog2_data
    )
    print(f"[MOG2][{name}] Completed!\n")


def annotateKnn(name, annotator, cap, learning_rate, dist2_threshold, detect_shadows, motion_energy_threshold, hysteresis, headless):
    print(f"[KNN][{name}] Running...")
    knn_data = knn(cap, learning_rate, dist2_threshold, detect_shadows, motion_energy_threshold, hysteresis, headless)
    annotator.add_motion_data(
        algorithm_name=f"KNN-{name}",
        parameters={
            "LearningRate": learning_rate,
            "DistanceToThreshold": dist2_threshold,
            "DetectShadows": detect_shadows,
            "MotionEnergyThreshold": motion_energy_threshold,
            "Hysteresis": hysteresis
        },
        motion_data=knn_data
    )
    print(f"[KNN][{name}] Completed!\n")