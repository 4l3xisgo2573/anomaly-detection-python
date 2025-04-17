# Asynchronous/Multithreading libraries
import asyncio
import threading

# Image processing/Camera libraries
import cv2
from picamera2 import Picamera2
from gi.repository import GLib

# Machine learning libraries
import torch
from torchvision.transforms import v2
import torch.nn as nn

# Data structuring libraries
import numpy as np
import json
import struct

# Bluetooth/System control libraries
import time
import RPi.GPIO as GPIO
from time import sleep
from bluezero import adapter, peripheral, device, async_tools

# Service and Characteristic UUIDs for Bluetooth
UART_SERVICE = '6E400001-B5A3-F393-E0A9-E50E24DCCA9E'
RX_CHARACTERISTIC = '6E400002-B5A3-F393-E0A9-E50E24DCCA9E'
TX_CHARACTERISTIC = '6E400003-B5A3-F393-E0A9-E50E24DCCA9E'
READ_CHAR = '2A6E'
# For some reason, RX was being troublesome, so we simply added a normal
# read characteristic to get data from the pi

# Setup GPIO pins 20 and 21 as PWM outputs
GPIO.setmode(GPIO.BCM)
GPIO.setup(20, GPIO.OUT)
GPIO.setup(21, GPIO.OUT)

# PWM frequency of 100Hz, starting at 55% pulse width
left = GPIO.PWM(20, 100)
right = GPIO.PWM(21, 100)
left.start(55)
right.start(55)

# Global variable to control when the robot runs autonomously vs manually
autonomous_mode = True

cv2.startWindowThread() # Deprecated?

# Choose and setup rasbpi camera to start taking images
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888',"size":(240,240)}))
picam2.start()

# Bluetooth buffer for sending images to the app
frame_buff = b'No image'

# Secondary thread to let bluetooth and camera functions run asynchronously
loop = asyncio.new_event_loop()
thr = threading.Thread(target=loop.run_forever, name='async runner')

# Returns the frame buffer to send over bluetooth
def read_value():
    global frame_buff
    
    to_send = list(frame_buff)
    return to_send

# Updates the TX characteristic with a new value
def update_value(characteristic): # Deprecated?
    new_value = read_value()
    characteristic.set_value(new_value)
    return new_value

# If the bluetooth client subscribes to a notifying characteristic, then
# start a timer to update the characteristic every few seconds
def notify_callback(notifying, characteristic):
    if notifying:
        async_tools.add_timer_seconds(2, update_value, characteristic)

# Set the mode of the robot to manual or autonomous
def toggle_mode():
    global autonomous_mode
    autonomous_mode = not autonomous_mode
    if autonomous_mode:
        mode = "Autonomous"
    else:
        mode = "Manual"
    print(f'Mode is {mode}')

# A class from the ble_uart.py example from the bluezero repository
class UARTDevice:
    tx_obj = None

    # Accepts bluetooth connection from client
    @classmethod
    def on_connect(cls, ble_device: device.Device):
        print("Connected to " + str(ble_device.address))

    # Completes disconect from bluetooth client
    @classmethod
    def on_disconnect(cls, adapter_address, device_address):
        print("Disconnected from " + device_address)

    # Assigns a characteristic as notifying
    @classmethod
    def uart_notify(cls, notifying, characteristic):
        if notifying:
            cls.tx_obj = characteristic
        else:
            cls.tx_obj = None

    # Sends a value through TX to client
    @classmethod
    def update_tx(cls):
        if cls.tx_obj:
            to_send = read_value()
            cls.tx_obj.set_value(to_send)
            return to_send

    # Reads a value through RX from client
    @classmethod
    def uart_write(cls, value, options):
        global autonomous_mode
        
        text = bytes(value).decode('utf-8')
        print('Text value:', text)
        
        # Toggles between manual and autonomous mode upon receiving the
        # message 'toggle'
        if text == 'toggle':
            toggle_mode()
        # Checks for various movement commands only if the robot isn't in
        # autonomous mode
        elif not autonomous_mode:
            # For reference, you can think of 55% PWM as stopped, 90% as
            # forward, and 30% as reverse. This code changes the speed of
            # the robot's motors depending on received bluetooth messages
            if text == 'stop':
                left.ChangeDutyCycle(55)
                right.ChangeDutyCycle(55)
            elif text == 'right':
                left.ChangeDutyCycle(90)
                right.ChangeDutyCycle(40)
            elif text == 'left':
                left.ChangeDutyCycle(40)
                right.ChangeDutyCycle(90)
            elif text == 'forward':
                left.ChangeDutyCycle(90)
                right.ChangeDutyCycle(90)
            elif text == 'reverse':
                left.ChangeDutyCycle(30)
                right.ChangeDutyCycle(30)

# A class to construct the ML model architecture we're using
class ConvolutionalAutoencoder(nn.Module):
        def __init__(self, num_layers=4, dropout_rate=0.2):
            super(ConvolutionalAutoencoder, self).__init__()
            
            # Create specifications of the encoder half of the autoencoder
            # Includes expanding channels through the encoder from 3 to 512
            encoder_layers = []
            in_channels = 3
            out_channels_list = [64, 128, 256, 512]
            out_channels_list = out_channels_list[:num_layers]
            
            # Adds a convolution, a normalization, and an activation
            # function for each of the out channel layers
            for out_channels in out_channels_list:
                encoder_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                ])
                in_channels = out_channels

            # Add a dropout layer for model effeciency
            encoder_layers.append(nn.Dropout(dropout_rate))

            # Package the created layers into an encoder
            self.encoder = nn.Sequential(*encoder_layers)

            # Create specifications of the decoder half of the autoencoder
            # Includes contracting channels through the decoder from 512 to 64
            decoder_layers = []
            out_channels_list.reverse()

            # Adds a convolution, a normalization, and an activation
            # function for each of the out channel layers
            for out_channels in out_channels_list[1:]:
                decoder_layers.extend([
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ])
                in_channels = out_channels
            
            # Add a final convolutional layer that can reproduce an image
            # with RGB pixels
            decoder_layers.append(nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.Sigmoid())

            # Package the created layers into a decoder
            self.decoder = nn.Sequential(*decoder_layers)

        # Propogates data from the input image through the encoder and
        # decoder to recreate an image
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

# Determines how the robot should move depending on where an anomaly appears
# in the input image
def get_movement_command(distance_x, distance_y):
    threshold = 20
    if abs(distance_x) < threshold and abs(distance_y) < threshold:
        return "STOP"
    if abs(distance_x) > abs(distance_y):
        return "RIGHT" if distance_x > 0 else "LEFT"
    return "REVERSE" if distance_y > 0 else "FORWARD"

# Loads the trained autoencoder for use on the robot
def load_trained_model():
    with open("best_params.json", "r") as f:
        best_params = json.load(f)
    model = ConvolutionalAutoencoder(
        num_layers=best_params["num_layers"],
        dropout_rate=best_params["dropout"]
    )

    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()
    return model, best_params

# Loads the trained autoencoer and saves it to the variable 'model'
model, _  = load_trained_model()
model.eval()

# Matrix transformations used to prepare an image for processing
transforms = v2.Compose([
    v2.ToPILImage(),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.ConvertImageDtype(torch.float32),
])

# Global variable used to create a timer for capturing images of anamolies
stop_start_time = None

# Capture an image using the capture_array() method
def capture_frame():
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    return frame

# Set up the bluetooth service
async def publish_bluetooth(adapter_address):
    ble_uart = peripheral.Peripheral(adapter_address, local_name='BLE UART')
    ble_uart.add_service(srv_id=1, uuid=UART_SERVICE, primary=True)
    
    # UART RX Characteristic for receiving commands from client
    ble_uart.add_characteristic(srv_id=1, chr_id=1, uuid=RX_CHARACTERISTIC,
                                value=[], notifying=False,
                                flags=['write', 'write-without-response'],
                                write_callback=UARTDevice.uart_write,
                                read_callback=None,
                                notify_callback=None)
    
    # UART TX Characteristic for transmitting messages to client
    # Deprecated?
    ble_uart.add_characteristic(srv_id=1, chr_id=2, uuid=TX_CHARACTERISTIC,
                                value=[], notifying=False,
                                flags=['notify'],
                                notify_callback=UARTDevice.uart_notify,
                                read_callback=None,
                                write_callback=None)
    
    # Simple Read Characteristic for transmitting messages to client
    ble_uart.add_characteristic(srv_id=1, chr_id=3, uuid=READ_CHAR,
                                 value=[], notifying=False,
                                 flags=['read', 'notify'],
                                 read_callback=read_value,
                                 write_callback=None,
                                 notify_callback=notify_callback)

    # Publish the bluetooth service, runs indefinitely
    ble_uart.on_connect = UARTDevice.on_connect
    ble_uart.on_disconnect = UARTDevice.on_disconnect
    ble_uart.publish()

# Setup the camera, start taking images, and process them
async def run_ml_model():
    global autonomous_mode, frame_buff
    
    # runs indefinitely
    while True:
        # Take an image through the camera and get it ready to be transmitted
        # to the app through bluetooth, and for ML processing
        frame = capture_frame()
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_tensor= transforms(frame).unsqueeze(0)
        frame_buff = buffer
        
        # Only process images and move the robot autonomously if in
        # autononmous mode
        if autonomous_mode:
            # Run the frame through the m odel
            with torch.no_grad():
                recon = model(frame_tensor)

            # Use loss from the autoencoder to determine image anomalies
            recon_error = ((frame_tensor - recon) ** 2).mean(dim=1).squeeze().numpy()
            recon_error = (recon_error * 255).astype(np.uint8)

            # Create heatmap of loss in input image
            heatmap = cv2.applyColorMap(recon_error, cv2.COLORMAP_JET)
            gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            _,thresholded = cv2.threshold(gray_heatmap, 100, 255, cv2.THRESH_BINARY)
            
            # Find regions of loss that pass a high enough threshold to be
            # something anomalous in the image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = heatmap.shape[:2]
            center_x, center_y = width // 2, height // 2
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]
            largest_contour = max(contours, key=cv2.contourArea, default=None)
            
            # If there is an anomaly, determine where it is in the image, and
            # move the robot to better view it
            if largest_contour is not None:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    # Determine the distance of the anomaly from the center
                    # of the image
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    distance_x = cX - center_x
                    distance_y = cY - center_y
                    movement_command = get_movement_command(distance_x, distance_y)
                    print(movement_command)

                    # For reference, you can think of 55% PWM as stopped, 90%
                    # as forward, and 30% as reverse. This code changes the
                    # speed of the robot's motors depending on the movement
                    # command determined from processing the image
                    if movement_command == 'STOP':
                        left.ChangeDutyCycle(55)
                        right.ChangeDutyCycle(55)
                    elif movement_command == 'RIGHT':
                        left.ChangeDutyCycle(90)
                        right.ChangeDutyCycle(40)
                    elif movement_command == 'LEFT':
                        left.ChangeDutyCycle(40)
                        right.ChangeDutyCycle(90)
                    elif movement_command == 'FORWARD':
                        left.ChangeDutyCycle(90)
                        right.ChangeDutyCycle(90)
                    elif movement_command == 'REVERSE':
                        left.ChangeDutyCycle(30)
                        right.ChangeDutyCycle(30)
                    
                    # If the robot is stopped in front of an anomaly for more
                    # than 4 seconds, take a photo of the anomaly
                    if movement_command == "STOP":
                        if stop_start_time is None:
                            stop_start_time = time.time()
                        elif time.time() - stop_start_time >= 4:
                            print('Capture')
                            stop_start_time = None
                    else:
                        stop_start_time = None

            #cv2.imshow("Camera", heatmap) # Deprecated?
        if cv2.waitKey(0) & 0xFF == ord('q'): # Deprecated?
            break

    # If user exits the window, stop the camera and distroy the windows
    picam2.stop()
    cv2.destroyAllWindows()

# Runs the bluetooth and anomaly detection tasks simultaneously
async def main(adapter_address):
    # Starts the thread
    if not thr.is_alive():
        thr.start()
    # Runs the anomaly detection in the created thread
    asyncio.run_coroutine_threadsafe(run_ml_model(), loop)
    # Runs the bluetooth service in this thread
    await publish_bluetooth(adapter_address)

if __name__ == '__main__':
    asyncio.run(main(list(adapter.Adapter.available())[0].address))
