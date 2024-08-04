import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

class VideoAnnotator:
    def __init__(self, video_file, output_folder):
        self.video_file = video_file
        self.output_folder = output_folder
        self.cap = cv2.VideoCapture(video_file)

        # Extract video metadata
        self.resolution = f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # self.encoding = self.get_video_encoding()
        self.duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.algorithms = []
        self.frames = []

        self.cap.release()

    def get_video_encoding(self):
        # Placeholder for actual encoding extraction method
        return "H264"

    def add_motion_data(self, algorithm_name, parameters, motion_data):
        algorithm_id = len(self.algorithms)
        # Store algorithm details
        algorithm_info = {
            "id": algorithm_id,
            "name": algorithm_name,
            "parameters": parameters
        }
        self.algorithms.append(algorithm_info)

        motion_energy_list = []
        motion_presence_list = []
        max_energy = 0
        # Store motion data for each frame
        for i, data in enumerate(motion_data):
            if i >= len(self.frames):
                self.frames.append({
                    "id_frame": i + 1,
                    "experiments": []
                })
            self.frames[i]["experiments"].append({
                "id": algorithm_id,
                "motion_energy": data['motion_energy'],
                "motion_presence": data['motion_presence'],
            })
            if (i > 5 and data['motion_energy'] > max_energy):
                max_energy = data['motion_energy']
            
            motion_energy_list.append(data['motion_energy'])
            motion_presence_list.append(1 if data['motion_presence'] else 0)
        
        # Plotting the motion data
        fig, ax = plt.subplots()
        ax.plot(motion_energy_list, label='Motion Energy')
        ax.plot([parameters["MotionEnergyThreshold"]] * len(motion_data), 'r--', label='Threshold')
        motion_presence_list = [((max_energy / 2) if ik == 1 else 0) for ik in motion_presence_list]
        ax.plot(np.array(motion_presence_list), label='Motion Presence')
        ax.set_xlim(0, len(motion_data))
        ax.set_ylim(0, max_energy + max_energy*0.1)
        ax.legend()
        plt.savefig(f"{self.output_folder}{algorithm_name}_motion_plot.png")
        plt.close()

    def save_to_json(self, target_filename):
        # Combine all information into a single dictionary
        video_info = {
            "video": self.video_file,
            "resolution": self.resolution,
            "fps": self.fps,
            # "encoding": self.encoding,
            "duration": self.duration,
            "num_frames": self.num_frames,
            "algorithms": self.algorithms,
            "frames": self.frames
        }

        # Save the dictionary as a JSON file
        with open(self.output_folder + target_filename, 'w') as f:
            json.dump(video_info, f, indent=4)
