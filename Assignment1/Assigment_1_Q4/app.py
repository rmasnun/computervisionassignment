from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Define the width of the reference object in centimeters
reference_width_cm = 10  # Change this to the actual width of the reference object

def object_measurement():
    # Start video capture
    cap = cv2.VideoCapture(0)  # Change index if using a different camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over detected contours
        for contour in contours:
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Measure object size (width) in pixels
            object_width_px = w

            # Convert object width from pixels to centimeters
            # Assuming camera is calibrated or a reference object with known width is present
            object_width_cm = (object_width_px / frame.shape[1]) * reference_width_cm

            # Draw bounding box and object width on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Width: {object_width_cm:.2f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame in bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release video capture
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(object_measurement(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
