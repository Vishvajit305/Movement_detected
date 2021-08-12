import cv2
import numpy as np

capture = cv2.VideoCapture(0)

ret, frame1= capture.read()
ret, frame2= capture.read()

while(capture.isOpened()):
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, thresh= cv2.threshold(blur,21,255,cv2.THRESH_BINARY)
    dialted = cv2.dilate(thresh, None, iterations=3)
    countours, _= cv2.findContours(dialted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame1, countours, -1, (255,0,0),3)

    for i in countours: 
        x,y,w,h =cv2.boundingRect(i)

        if cv2.contourArea(i)<1000:
            continue
        cv2.rectangle(frame1, (x,y),(x+w,y+h), (0,0,255),3)
        cv2.putText(frame1, f"Status: {'MOVEMENT DETECTED'}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),3)

    cv2.imshow("sample",frame1)
    frame1=frame2
    ret, frame2=capture.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
