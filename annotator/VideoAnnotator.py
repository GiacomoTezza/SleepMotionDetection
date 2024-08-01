import cv2
import json

class VideoAnnotator:
    def __init__(self, video_file):
        self.video_file = video_file
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
        with open(target_filename, 'w') as f:
            json.dump(video_info, f, indent=4)

# Example usage
if __name__ == "__main__":
    annotator = VideoAnnotator('path_to_video.mp4')
    
    # Example motion data from MOG2 algorithm
    example_motion_data = [
        {"motion_present": True, "motion_energy": 0.1},
        {"motion_present": False, "motion_energy": 0.0},
        # Add more frames as needed
    ]
    
    annotator.add_motion_data(
        algorithm_id=1,
        algorithm_name="MOG2-parameters-1",
        parameters={
            "LearningRate": 0.1,
            "History": 10,
            "VarThreshold": 10
        },
        motion_data=example_motion_data
    )
    
    annotator.save_to_json('output.json')
