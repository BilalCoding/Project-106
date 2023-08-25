import cv2

body_classifier = cv2.CascadeClassifier("C:/Users/bilal/Desktop/Coding Files/Projects/106/haarcascade_fullbody.xml")

cap = cv2.VideoCapture("C:/Users/bilal/Desktop/Coding Files/Projects/106/walking.avi")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
