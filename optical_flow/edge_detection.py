import cv2

video = "/home/iain/Desktop/HonoursProject/optical_flow/IMG_8052.MOV"
cap = cv2.VideoCapture(video)
while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 70)
    ret, mask = cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("Video feed", mask)

    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()
