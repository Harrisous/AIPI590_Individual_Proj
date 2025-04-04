''' 
AIPI590-10 AI in physical world
Individual Project: Auto Flame Detector and Blower. 

It can be used to mail to friends far away to celebrate their birthday remotely and asynchronously.

The device will use CV to detect fire and a temperature sensor to verify the temperature, then blow out the fire with a spinning fan. 
The physical input would be a user input to set the age with a potentiometer, and the age will be shown on an LCD screen.
A CV model will be used for tracking the fire, and a language model API may be added to say "happy xx th birthday". 

Model source: 
yolov8s_best.pt (inference time: 3s) https://github.com/Yusuf-ozen/Yolov8_Fire_Detection.git
yolov8s_.pt (inference time: 1s) https://github.com/salim4n/Fire_Detection.git

Wiring: https://app.cirkitdesigner.com/project/62b2e3c6-e88a-4f61-8328-c9522a157c0a
'''
import RPi.GPIO as GPIO
from lcd_i2c import LCD_I2C
import smbus
import time
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import os
from dotenv import load_dotenv
from playsound import playsound
from pathlib import Path
from openai import OpenAI
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
factory = PiGPIOFactory()# Requires running `sudo pigpiod`
# load open credentials
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# picam config
desired_fps = 30
frame_duration = int(1e6 / desired_fps)
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
config["controls"] = {"FrameDurationLimits": (frame_duration, frame_duration)}
picam2.configure(config)
picam2.start()

class LcdDevice:
    '''
        <LCD screen> display age
    '''
    def __init__(self, DEVICE_ADDRESS = 0x27, width=16, length=2):
        lcd = LCD_I2C(DEVICE_ADDRESS, width, length)
        lcd.backlight.on()
        lcd.blink.off()
        self.lcd = lcd

    def update_age(self, age=None):
        '''Update the speed number with new distance and new speed'''
        # update display
        self.lcd.clear()
        self.lcd.cursor.setPos(0, 4)
        self.lcd.write_text(f"Age: {age}")

    def cleanup(self):
        self.lcd.backlight.off()
        self.lcd.clear()


class AgeReader:
    '''
        <age adjustor> analog to digital converter, setting age from position 
    '''
    def __init__(self, I2C_BUS = 1, DEVICE_ADDRESS = 0x4b):
        self.bus = smbus.SMBus(I2C_BUS) # ADC address -> bus obj
        self.DEVICE_ADDRESS = DEVICE_ADDRESS
    
    def read_age(self, channel = 0):
        '''return: age <int> between 0-255'''
        control_byte = 0x84 | (channel << 4)
        self.bus.write_byte(self.DEVICE_ADDRESS, control_byte)
        age = self.bus.read_byte(self.DEVICE_ADDRESS)
        age = int(age/255*100)
        return age

    def close(self):
        self.bus.close()

class DCMotor:
    '''
        <DC motor> dc motor to put out fire
    '''
    def __init__(self, DC_ENA = 18, DC_IN1 = 17, DC_IN2 = 27):
        self.DC_ENA = DC_ENA
        self.DC_IN1 = DC_IN1
        self.DC_IN2 = DC_IN2

        GPIO.setup(self.DC_ENA, GPIO.OUT)
        GPIO.setup(self.DC_IN1, GPIO.OUT)
        GPIO.setup(self.DC_IN2, GPIO.OUT)

        # set dc motor to forward state
        GPIO.output(self.DC_IN1, GPIO.LOW)
        GPIO.output(self.DC_IN2, GPIO.HIGH)

        self.pwm = GPIO.PWM(self.DC_ENA, 100) # ENA set up: 100hz
        self.pwm.start(0) # start from 0% duty cycle
    
    def update_speed(self, new_speed):
        '''new speed: int, from 0-100'''
        self.pwm.ChangeDutyCycle(new_speed) # more accurate transition

    
def generate_and_play_audio(age, voice="coral"):
    # generate prompt 
    print("generate text...")
    prompt = f"It's my friends birthday, the age turns to {age}; generate a short words to celebrate. Generate only text."
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    text = completion.choices[0].message.content
    print("text",text)
    print("generating sound..")
    # Define the path for the audio file
    speech_file_path = Path(__file__).parent / "speech.mp3"
    
    # Generate speech using OpenAI's streaming response API
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input="heyyy! "+text,
        instructions="Speak in a cheerful and friendly tone.",
    ) as response:
        response.stream_to_file(speech_file_path)
    print("playing sound...")
    # Play the generated audio file through the USB speaker
    playsound(str(speech_file_path))


''' testing functions '''
def test_age_input_and_display(lcd, age_reader):
    while True:
        age = age_reader.read_age()
        lcd.update_age(age)
        time.sleep(0.1)

def test_motor(dc_motor):
    dc_motor.update_speed(100)
    time.sleep(1)
    dc_motor.update_speed(0)

def test_servo(servo_yaw, servo_pitch):
    servos = [servo_yaw, servo_pitch]
    for servo in servos:
        servo.min()
        print("Now at: ",servo.value)
        time.sleep(1)
        for i in range(20):
            servo.value = (i-10)/10
            print("Now at: ",servo.value)
            time.sleep(1)
        servo.mid()


'''entry point'''
if __name__ == "__main__":
    # remember to run `sudo pigpiod` before running the code
    # initialize all devices
    lcd = LcdDevice(DEVICE_ADDRESS = 0x27) # lcd screen for age display
    age_reader = AgeReader() # for input ages
    dc_motor = DCMotor() # to blow out fire when
    servo_yaw = Servo(12, pin_factory=factory, 
              min_pulse_width=0.5/1000,  # 0.5ms
              max_pulse_width=2.5/1000)
    servo_pitch = Servo(13, pin_factory=factory, 
                min_pulse_width=0.5/1000,  
                max_pulse_width=2.5/1000)

    servo_yaw.mid()
    servo_pitch.mid()
    
    # initialize the model
    print("Loading model...")
    # model = YOLO('yolov8s_best.pt')
    model = YOLO('fire_model.pt') 
    frame_width = picam2.camera_configuration()["main"]["size"][0]
    frame_height = picam2.camera_configuration()["main"]["size"][1]
    frame_center = (frame_width//2, frame_height//2)
    YAW_SCALE = 0.2/frame_width  # 0.2 full scale per frame width
    PITCH_SCALE = 0.15/frame_height
    try:
        # age input
        decision_time = 3
        for i in range(decision_time*100):
            age = age_reader.read_age()
            lcd.update_age(age)
            time.sleep(0.01)
        
        # happy birthday speech
        generate_and_play_audio(age)

        # blow out the fire
        while True:
            frame = picam2.capture_array()  # live frame
            frame = cv2.flip(frame, -1) # inverse the frame (due to hardware setting)
            frame_RGB =  cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB) # keep only RGB
            results = model(frame_RGB, imgsz=640, conf=0.4)
            print(results)
            if len(results[0].boxes) == 0:
                print("No fire detected...")
                dc_motor.update_speed(0)
            else:
                dc_motor.update_speed(100)
                # Get the first detected fire's bounding box
                box = results[0].boxes.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                fire_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)  # Center of the box

                # Calculate error and move the camera
                error_x = fire_center[0] - frame_center[0]
                error_y = fire_center[1] - frame_center[1]
                print("error_x: ", error_x)
                print("error_y: ", error_y)
                new_yaw = servo_yaw.value - error_x * YAW_SCALE
                new_pitch = servo_pitch.value - error_y * PITCH_SCALE

                print("yaw:", new_yaw)
                print("new_pitch:", new_pitch)
                servo_yaw.value = np.clip(new_yaw, -0.9, 0.9)
                servo_pitch.value = np.clip(new_pitch, -0.9, 0.4)

    except KeyboardInterrupt:
        print("\n Exiting Program")
    except Exception as e:
        print(e)
    finally:
        servo_yaw.mid()
        servo_pitch.mid()
        picam2.stop()
        cv2.destroyAllWindows()
        age_reader.close()
        lcd.cleanup()