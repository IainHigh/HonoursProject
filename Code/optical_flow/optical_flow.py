import numpy as np
import cv2 as cv

# video = (
#     "/home/iain/Desktop/HonoursProject/optical_flow/production_id_4034132 (2160p).mp4"
# )

# video = "/home/iain/Desktop/HonoursProject/optical_flow/IMG_8052.MOV"

# video = "/home/iain/Desktop/HonoursProject/optical_flow/video (2160p).mp4"

video = "/home/iain/Desktop/HonoursProject/optical_flow/pexels_videos_4103 (1080p).mp4"

video = (
    "/home/iain/Desktop/HonoursProject/optical_flow/production_id_3831903 (2160p).mp4"
)

video = (
    "/home/iain/Desktop/HonoursProject/optical_flow/production_id_4887478 (2160p).mp4"
)

cap = cv.VideoCapture(video)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=100, blockSize=1)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_frame = cv.resize(old_frame, (1280, 720))
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while 1:
    ret, frame = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    frame = cv.resize(frame, (1280, 720))

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(frame_gray, (5, 5), 0)
    canny = cv.Canny(blur, 70, 100)
    _, boundary_mask = cv.threshold(canny, 0, 255, cv.THRESH_BINARY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv.add(frame, mask)

    # Overlay the boundary mask on the frame with optical flow
    combined_image = cv.add(img, cv.cvtColor(boundary_mask, cv.COLOR_GRAY2BGR))

    cv.imshow("combined_image", combined_image)

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
