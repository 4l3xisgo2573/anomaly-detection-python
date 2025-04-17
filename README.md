# Rasberry Pi Autonomous Robot with Anomaly Detection and Bluetooth Control

## Features

- Live Camera Feed
- Machine Learning Detection
- Bluetooth Communication
- Dual Mode Control (Autonomous and Manual)
- Motor Control

## Technologies Used

- Python
- Asyncio and Threading
- OpenCV
- Torch
- Numpy
- Bluezero
- RPi.GPIO
- Picamera2

## Running the Program

```markdown
python3 main.py
```

The robot will start in **Autonomous Mode** by default. You can toggle to **Manual Mode** by sending the toggle command via Bluetooth

In Manual Mode, supported commands include:

- forward
- reverse
- left
- right
- stop

## Machine Learning Model

The robot uses a pre-trained **Convolutional Autoencoder** to detect visual anomalies in camera frame
Model loads from:

- trained_model.pth
- best_params.json

## Robot Logic

In **Autonomous Mode**, the robot:

- Captures a frame
- Passes it through the autoencoder
- Computes the reconstruction error
- Uses a heatmap to find anomolies
- Adjusts motor movement based on the postion of the detected anomaly

In **Manual Mode**, the robot:

- Awaits movement commands via Bluetooth
- Adjust PWM duty cyles to control motion
