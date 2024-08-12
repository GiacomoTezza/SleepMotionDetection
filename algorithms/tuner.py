import itertools
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from .knn import knn
from .mog import mog1, mog2
from annotator import annotateMog1, annotateMog2, annotateKnn

def calculate_consistency(motion_data):
    # Calculate consistency as the proportion of consecutive frames with similar motion presence
    motion_presence = [frame['motion_presence'] for frame in motion_data]
    consistency_score = sum(1 for i in range(1, len(motion_presence)) if motion_presence[i] == motion_presence[i-1])
    return consistency_score / (len(motion_presence) - 1)

def calculate_motion_energy_distribution(motion_data):
    # Calculate variance of motion energy to assess the distribution
    motion_energies = [frame['motion_energy'] for frame in motion_data]
    return np.var(motion_energies), np.mean(motion_energies)

def calculate_noise_level(motion_data):
    # Calculate noise level as the proportion of very short-duration motion events
    short_motion_events = 0
    motion_presence = [frame['motion_presence'] for frame in motion_data]
    event_length = 0

    for i in range(1, len(motion_presence)):
        if motion_presence[i]:
            event_length += 1
        else:
            if event_length > 0 and event_length < 5:  # Short events < 5 frames
                short_motion_events += 1
            event_length = 0

    noise_score = short_motion_events / len(motion_data)
    return noise_score


def run_mog1(params):
    lr, h, nm, br, ns, met, hys, video_path, headless = params
    cap = cv2.VideoCapture(video_path)
    mog1_data = mog1(cap, lr, h, nm, br, ns, met, hys, headless=headless, show_progress=False)
    consistency_score = calculate_consistency(mog1_data)
    variance, mean_motion_energy = calculate_motion_energy_distribution(mog1_data)
    noise_score = calculate_noise_level(mog1_data)
    total_score = (consistency_score * 0.4) + (1 / (variance + 1) * 0.3) - (noise_score * 0.3)

    return {
        "parameters": {
            "LearningRate": lr,
            "History": h,
            "NumberMixtures": nm,
            "BackgroundRatio": br,
            "NoiseSigma": ns,
            "MotionEnergyThreshold": met,
            "Hysteresis": hys
        },
        "scores": {
            "consistency": consistency_score,
            "variance": variance,
            "mean_motion_energy": mean_motion_energy,
            "noise_score": noise_score,
            "total_score": total_score
        },
        "motion_data": mog1_data
    }


def run_mog2(params):
    lr, met, hys, video_path, headless = params
    cap = cv2.VideoCapture(video_path)
    mog2_data = mog2(cap, lr, met, hys, headless=headless, show_progress=False)
    consistency_score = calculate_consistency(mog2_data)
    variance, mean_motion_energy = calculate_motion_energy_distribution(mog2_data)
    noise_score = calculate_noise_level(mog2_data)
    total_score = (consistency_score * 0.4) + (1 / (variance + 1) * 0.3) - (noise_score * 0.3)

    return {
        "parameters": {
            "LearningRate": lr,
            "MotionEnergyThreshold": met,
            "Hysteresis": hys
        },
        "scores": {
            "consistency": consistency_score,
            "variance": variance,
            "mean_motion_energy": mean_motion_energy,
            "noise_score": noise_score,
            "total_score": total_score
        },
        "motion_data": mog2_data
    }


def run_knn(params):
    lr, d2t, ds, met, hys, video_path, headless = params
    cap = cv2.VideoCapture(video_path)
    knn_data = knn(cap, lr, d2t, ds, met, hys, headless=headless, show_progress=False)
    consistency_score = calculate_consistency(knn_data)
    variance, mean_motion_energy = calculate_motion_energy_distribution(knn_data)
    noise_score = calculate_noise_level(knn_data)
    total_score = (consistency_score * 0.4) + (1 / (variance + 1) * 0.3) - (noise_score * 0.3)

    return {
        "parameters": {
            "LearningRate": lr,
            "DistanceToThreshold": d2t,
            "DetectShadows": ds,
            "MotionEnergyThreshold": met,
            "Hysteresis": hys
        },
        "scores": {
            "consistency": consistency_score,
            "variance": variance,
            "mean_motion_energy": mean_motion_energy,
            "noise_score": noise_score,
            "total_score": total_score
        },
        "motion_data": knn_data
    }

def grid_search_mog1(annotator, video_path):
    learning_rates = [-1]
    histories = [100, 200, 300]
    n_mixtures = [3, 5, 7]
    background_ratios = [0.05, 0.1, 0.2]
    noise_sigmas = [0.5, 1, 1.5]
    motion_energy_thresholds = [0.01, 0.015, 0.002]
    hysteresis_values = [10, 25, 50]
    total_combinations = len(learning_rates) * len(histories) * len(n_mixtures) * len(background_ratios) * len(noise_sigmas) * len(motion_energy_thresholds) * len(hysteresis_values)

    params = list(itertools.product(
        learning_rates, histories, n_mixtures, background_ratios, noise_sigmas, motion_energy_thresholds, hysteresis_values, [video_path], [True]))

    with Pool(processes=4) as pool:  # Use multiprocessing to speed up the computation
        results = list(tqdm(pool.imap(run_mog1, params), total=total_combinations))

    # Sort results by total score in descending order
    sorted_results = sorted(results, key=lambda x: x['scores']['total_score'], reverse=True)

    # Save and inspect the top N configurations
    top_n = 5
    for i, result in enumerate(sorted_results[:top_n]):
        param_name = f"Top_{i+1}_Score_{result['scores']['total_score']:.2f}"
        annotator.add_motion_data(
            algorithm_name=f"MOG1-{param_name}",
            parameters=result['parameters'],
            motion_data=result['motion_data'],
            notes=result['scores']
        )


def grid_search_mog2(annotator, video_path):
    learning_rates = [-1]
    motion_energy_thresholds = [0.01, 0.015, 0.002]
    hysteresis_values = [10, 25, 50]
    total_combinations = len(learning_rates) * len(motion_energy_thresholds) * len(hysteresis_values)

    params = list(itertools.product(
        learning_rates, motion_energy_thresholds, hysteresis_values, [video_path], [True]))

    with Pool(processes=4) as pool:  # Use multiprocessing to speed up the computation
        results = list(tqdm(pool.imap(run_mog2, params), total=total_combinations))

    # Sort results by total score in descending order
    sorted_results = sorted(results, key=lambda x: x['scores']['total_score'], reverse=True)

    # Save and inspect the top N configurations
    top_n = 5
    for i, result in enumerate(sorted_results[:top_n]):
        param_name = f"Top_{i+1}_Score_{result['scores']['total_score']:.2f}"
        annotator.add_motion_data(
            algorithm_name=f"MOG2-{param_name}",
            parameters=result['parameters'],
            motion_data=result['motion_data'],
            notes=result['scores']
        )


def grid_search_knn(annotator, video_path):
    learning_rates = [-1]
    distance_to_threshold = [200, 400, 600]
    detect_shadows = [True, False]
    motion_energy_thresholds = [0.01, 0.015, 0.002]
    hysteresis_values = [10, 25, 50]
    total_combinations = len(learning_rates) * len(distance_to_threshold) * len(detect_shadows) * len(motion_energy_thresholds) * len(hysteresis_values)

    params = list(itertools.product(
        learning_rates, distance_to_threshold, detect_shadows, motion_energy_thresholds, hysteresis_values, [video_path], [True]))

    with Pool(processes=4) as pool:  # Use multiprocessing to speed up the computation
        results = list(tqdm(pool.imap(run_knn, params), total=total_combinations))

    # Sort results by total score in descending order
    sorted_results = sorted(results, key=lambda x: x['scores']['total_score'], reverse=True)

    # Save and inspect the top N configurations
    top_n = 5
    for i, result in enumerate(sorted_results[:top_n]):
        param_name = f"Top_{i+1}_Score_{result['scores']['total_score']:.2f}"
        annotator.add_motion_data(
            algorithm_name=f"KNN-{param_name}",
            parameters=result['parameters'],
            motion_data=result['motion_data'],
            notes=result['scores']
        )