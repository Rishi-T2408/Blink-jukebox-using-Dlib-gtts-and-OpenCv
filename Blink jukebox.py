import cv2
import numpy as np
import dlib
from math import  hypot
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play


def play_audio(path_to_audio):
    music = AudioSegment.from_mp3(path_to_audio)
    play(music)

text="Welcome to Blinking Jukebox..... " \
     "Opening virtual keyboard for you..... " \
     "Please blink our eye to select any song"
my_text = gTTS(text)
my_text.save('hello.mp3')
play_audio("hello.mp3")



detector =dlib.get_frontal_face_detector() #object for detecting face
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# keyboard settings
keyboard=np.zeros((600,600,3),dtype=np.uint8)
keys_set_1={0:"Namo Namo",1:"Chlorine",2:"All Of The Stars"}


def letter(letter_index,text,letter_light):
    if letter_index==0:
        x=0
        y=0
    elif letter_index == 1:
        x=0
        y=200
    elif letter_index == 2:
        x=0
        y=400


    width=600
    height=200
    th=2
    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard,(x+th,y+th),(x+width-th,y+height-th),(0,255,0),th)

    # Text setting
    font_letter=cv2.FONT_HERSHEY_PLAIN
    font_scale=4
    font_th=4
    text_size=cv2.getTextSize(text,font_letter,font_scale,font_th)[0]

    width_text , height_text = text_size[0] ,text_size[1]
    text_x=int((width-width_text)/2)+x
    text_y=int((height+height_text)/2)+y

    cv2.putText(keyboard,text,(text_x,text_y),font_letter,font_scale,(255,0,0),font_th)


def midpoint(p1,p2):
    return  int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)



cap=cv2.VideoCapture(0)
board=np.zeros((800,800),dtype=np.uint8)
board[:]=255
# using dlib we can detect 68 facial marks point

font=cv2.FONT_HERSHEY_COMPLEX_SMALL

def get_blinking_ratio(eye_points,facial_landmarks):
    left_point = ((facial_landmarks.part(eye_points[0]).x), (facial_landmarks.part(eye_points[0]).y))
    right_point = ((facial_landmarks.part(eye_points[3]).x), (facial_landmarks.part(eye_points[3]).y))
    centre_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    centre_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, centre_top, centre_bottom, (200, 0, 50), 2)
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[0] - right_point[0]))
    ver_line_lenght = hypot((centre_top[0] - centre_bottom[0]), (centre_top[1] - centre_bottom[1]))
    # print(hor_line_lenght/ver_line_lenght)
    # y = landmarks.part(36).y
    # showing the position of the 36th point of facial landmarks
    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def get_gaze_ratio(eye_points,facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
                                ], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    # cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y:max_y, min_x:max_x]

    # gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_eye_threshold = threshold_eye[0: height, 0:int(width / 2)]
    left_side_white = cv2.countNonZero(left_eye_threshold)
    right_eye_threshold = threshold_eye[0: height, int(width / 2):width]
    right_side_white = cv2.countNonZero(right_eye_threshold)

    if left_side_white==0:
        gaze_ratio=1
    elif right_side_white==0:
        gaze_ratio=5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


# Counter
frames = 0
letter_index=0
blinking_frames=0
text=""

while True:
    _, frame=cap.read()
    frame=cv2.resize(frame,None,fx=0.8,fy=0.8)
    keyboard[:] = (0,0,0)
    frames+=1
    new_frame = np.zeros((500, 500, 3), dtype=np.uint8)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    active_letter = keys_set_1[letter_index]
    faces=detector(gray)
    for face in faces:
        x,y = face.left(),face.top()
        x1,y1=face.right(),face.bottom()
        cv2.rectangle(frame,(x,y),(x1,y1),(0,0,0),2)
        facial_landmarks=predictor(gray, face)
        #cv2.circle(frame,(x,y),3,(255,255,255),2)

        #detect blinkng
        left_eye_ratio = get_blinking_ratio((36,37,38,39,40,41), facial_landmarks)
        right_eye_ratio=get_blinking_ratio((42,43,44,45,46,47), facial_landmarks)
        blinking_ratio= (left_eye_ratio+right_eye_ratio)/2
        if blinking_ratio >5.7:
            cv2.putText(frame,"BLINKING",(50,150),font,3,(50,0,255))
            blinking_frames+=1
            frames-=1
            if blinking_frames==5:
               text+=active_letter
               print(active_letter)
               play_audio(active_letter+".mp3")

        else:
            blinking_frames=0




        #gaze detection
        gaze_ratio_left_eye = get_gaze_ratio((36,37,38,39,40,41), facial_landmarks)
        gaze_ratio_right_eye = get_gaze_ratio((42,43,44,45,46,47), facial_landmarks)
        gaze_ratio=((gaze_ratio_right_eye) + (gaze_ratio_left_eye))/2

        cv2.putText(frame,str(gaze_ratio),(50,100),font,2,(132,255,138),3)

        if gaze_ratio<=1:
            cv2.putText(frame, "RIGHT", (250, 300), font, 2 , (2, 255, 1), 3)
            new_frame[:] = (0,0,255)
        elif 1<gaze_ratio<2.2:
            cv2.putText(frame,"CENTRE", (250, 300), font, 2 ,(2, 255, 1), 3)
        else:
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "LEFT", (250, 300), font, 2, (2, 255, 1), 3)

        #threshold_eye = cv2.resize(threshold_eye, None, fx=3, fy=3)
        #eye = cv2.resize(gray_eye, None, fx=3, fy=3)
        #cv2.imshow("Threshold",threshold_eye)
        #cv2.imshow("left",left_eye_threshold)
        #cv2.imshow("right",right_eye_threshold)
    #letters
    if frames==15:  #huristics other than machine learning......
        letter_index+=1
        frames=0
    if letter_index==3:
        letter_index=0

    for i in range(0, 3):                         #pyautoguiPlease blink our eye to select any song
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, keys_set_1[i], light)


    cv2.putText(board, text, (10, 100), font, 3, 0, 3)
    cv2.imshow("Eye movement platform",frame)
    #cv2.imshow("New frame", new_frame)

    cv2.imshow("Virtual keyboard",keyboard)
    cv2.imshow("Board",board)
    key =cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
