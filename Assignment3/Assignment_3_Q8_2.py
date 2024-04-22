import cv2

# Open the video camera
cap = cv2.VideoCapture(0)

# Create the CSRT Tracker
tracker = cv2.TrackerCSRT_create()

# Initialize the bounding box
bbox = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale (for better tracking performance)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create bounding box if not created
    if bbox is None:
        bbox = cv2.selectROI("Object Tracker", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, bbox)
    
    # Update the tracker
    ret, bbox = tracker.update(frame)
    
    # Draw the bounding box
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    
    # Display tracker
    cv2.imshow("Object Tracker", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()
