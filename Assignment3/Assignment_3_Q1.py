import cv2
import numpy as np
import random

# Capture video
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))

# Record 10 seconds of video
duration = 10  # in seconds
start_time = cv2.getTickCount()
while True:
    ret, frame = cap.read()
    out.write(frame)
    cv2.imshow('Frame', frame)
    if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() >= duration:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Read the saved video
video = cv2.VideoCapture('output.mp4')

# Read one frame from the video
_, template_frame = video.read()

# Select a region of interest from the frame
x, y, w, h = cv2.selectROI(template_frame)
template = template_frame[y:y+h, x:x+w]

# Set the method to compare template and search image
method = cv2.TM_CCOEFF_NORMED

# Compare template with randomly selected frames from the video
for _ in range(10):
    video.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT))))
    _, search_frame = video.read()
    search_img = search_frame[y:y+h, x:x+w]
    
    res = cv2.matchTemplate(search_img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(search_frame, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Detected', search_frame)
    cv2.waitKey(1000)

video.release()
cv2.destroyAllWindows()
