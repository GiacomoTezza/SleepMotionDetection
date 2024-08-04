# SleepMotionDetection
Motion Detection and Analysis for Sleep Monitoring Videos

## Objective:
The goal of this project is to develop a motion detection algorithm and analyze videos of people sleeping to identify frames with motion and measure the degree of motion.
This task offers an opportunity to gain hands-on experience in computer vision and data analysis while contributing to the development of sleep monitoring technology.

## Tasks:
### Algorithm research and implementation:
- [X] Research and select a suitable motion detection algorithm (MOG1, MOG2, KNN).
- [ ] Checkout papers "Action Recognition with Dynamic Image Networks" https://arxiv.org/abs/1612.00738. if interested in exploring more advanced techniques for motion detection.
- [X] Implement the chosen algorithm using Python and OpenCV. Can be more than one algorithm.
- [X] Understand the functioning of the chosen algorithm and its parameters.
### Parameters Tuning and Motion Energy Measurement:
- [ ] Identify the parameters affecting the sensitivity to motion.
- [ ] Experiment with different parameter values to optimize motion detection.
- [X] Develop a method to measure the "motion energy" for each frame.
### Handling False Positives :
- [ ] Investigate and address cases of false positives, such as motion caused by video artifacts.
- [ ] Implement strategies to minimize false detections and ensure accurate results.
### Testing and deliverables :
- [X] Python script implementing the motion detection algorithm.
- [ ] Analyze the motion detection results for each video and create tech report detailing the parameter tuning process and findings.
- [X] Dataset containing motion energy and motion presence data for each frame. The idea is to generate a structured data file (preferably in CSV or JSON format) for each video, containing information about motion energy and the presence of motion for every frame based on the chosen parameters.
- [ ] Optional: create a visualization tool to display the motion detection results and motion energy over time (simply web-based or using a Python library like Matplotlib).

#### Target data file example:
```
{
    "video": "video_1",
    "resolution": "1920x1080",
    "fps": 25,
    "encoding": "H264",
    "algorithms":
    [
        {
            "id": 1,
            "name": "MOG2-parameters-1",
            "parameters":
            {
                "LearningRate": 0.1,
                "History": 10,
                "VarThreshold": 10
            }
        },
        {
            "id": 2,
            "name": "MOG2-parameters-2",
            "parameters":
            {
                "LearningRate": 0.1,
                "History": 5,
                "VarThreshold": 10
            }
        },
        {
            "id": 3,
            "name": "Action Recognition",
            "parameters":
            {
                ???
            }
        },
    ]
    "frames":
    [
        {
            "id_frame": 1,
            "artifacts": False,
            "note": "H264 compression artifacts not detected",
            "experiments":
            [
                {
                    "id": 1,
                    "motion_energy": 0.1,
                    "motion_presence": False,
                },
                {
                    "id": 2,
                    "motion_energy": 0.5,
                    "motion_presence": True,
                }
            ]
        }
    ]
}
```

## Additional Information:
The motion detection algorithm should be efficient and scalable for analyzing large volumes of video data.
Experimentation and parameter tuning are crucial for achieving accurate motion detection results.
We encourage the use of Python and OpenCV for this task, but you are free to explore other tools and libraries if necessary.
Consider to create a modular and reusable codebase that can be extended for future enhancements and applications.

## Requirements
- Python (3.11 suggested since dependencies are not updated for 3.12)

## How to run
Create the virtual enviroment:
```
python -m venv venv
```
Activate the virtual enviroment:
```
source venv/bin/activate
```
Install requirements:
```
pip install -r requirements.txt
```
Run the program:
```
python3 main.py
```
To exit from the virtual enviroment:
```
deactivate
```
