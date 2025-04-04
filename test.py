import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Load the YOLO model
print("Loading model...")
# model = YOLO('yolov8s_best.pt')  
model = YOLO('fire_model.pt')
try:
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)  # Flip the frame if necessary (hardware-specific)
        frame_RGB =  cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        # Run the YOLO model on the frame
        results = model(frame_RGB, imgsz=640, conf=0.3)

        # Annotate the frame with the model's predictions
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Fire Detection Test", annotated_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()