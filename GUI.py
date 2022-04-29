import PySimpleGUI as sg
import cv2
import numpy as np
import os
import math
import time

xx=[]; yy=[]

#----- Функция нахождения СТП из попдрограммы нахождения попаданий -----
def ras(CrZX, CrZY):

    CrZX = ((float(xx[0]) + float(xx[1]) + float(xx[2])) / 3)
    CrZY = ((float(yy[0]) + float(yy[1]) + float(yy[2])) / 3)
    CrZX = float(CrZX) / 26
    CrZY = float(CrZY) / 20

    def toFixed(numObj, digits=0):
        return f"{numObj:.{digits}f}"

    CrZY = float(toFixed(CrZY, 2))
    CrZX = float(toFixed(CrZX, 2))
    print('СТП x= ', CrZX, ' y= ', CrZY)
    if CrZX > 0:
        print('На ', math.fabs(CrZX), ' влево')
    else:
        print('На ', math.fabs(CrZX), ' вправо')
    if CrZY > 0:
        print('На ', math.fabs(CrZY), ' оборотов против часовой')
    else:
        print('На ', math.fabs(CrZY), ' оборотов по часовой')

#----- Функция работы SIFT ----
def sift_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return len(good_matches)

#----- Основная функция -----
def main():
    #------ Радел интерфейса ------
    sg.theme('dark')
    layout = [[sg.Image(filename='', key='image')],
              [sg.Output(s=(109, 5))],
              [sg.Button('Стрельба', size=(10, 1), font='Helvetica 14'),
               sg.Button('Стоп', size=(10, 1), font='Any 14'),
               sg.Button('Калибровка', size=(10, 1), font='Helvetica 14'),
               sg.Button('Выход', size=(10, 1), font='Helvetica 14'), ]]

    window = sg.Window('Программное обеспечение "Кондор"',
                       layout, location=(800, 400))

    #------ Основные настройки камеры, передача кадров в видеопотоке -----
    cap = cv2.VideoCapture(0)
    cap.set(3, 780)
    cap.set(4, 520)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    height, width = frame1.shape[:2]
    i = 0
    token = 0
    start = False
    sift = False


    #----- Тело процесса -----
    while True:
        event, values = window.read(timeout=20)
        if event == 'Выход' or event == sg.WIN_CLOSED:
            return

        elif event == 'Стрельба':
            start = True
            sift = False

        elif event == 'Стоп':
            start = False
            sift = False


        elif event == 'Калибровка':
            sift = True
            start = False

        #----- Подпрограмма нахождения попаданий -----
        if start:
            diff = cv2.absdiff(frame1,
                               frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None,
                                 iterations=3)
            сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for contour in сontours:
                (x, y, w, h) = cv2.boundingRect(
                    contour)
                cv2.putText(frame1, "Status:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                            cv2.LINE_AA)

                if cv2.contourArea(contour) < 1000:
                    continue
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0),
                              2)
                cv2.putText(frame1, "Status: {}".format("Hit"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                            cv2.LINE_AA)

                #print('Попадание ', x, y, cv2.contourArea(contour))
                xc = (((x + (x + w)) / 2)/3) - ((width/2)/3)
                xx.append(xc)
                yc = (((y + (y + h)) / 2)/2) - ((height/2)/2)
                yy.append(yc)
                i+=1

                if i == 3:
                    ras(xx, yy)
                    i = 0
                    start = False

        #----- Подпрограмма управления камерой и каллибровки -----
        if sift:
            image_template = cv2.imread('img/test.jpg', 0)
            top_left_x = int(width / 3)
            top_left_y = int(height)
            bottom_right_x = int((width / 3) * 2)
            bottom_right_y = 0
            cv2.rectangle(frame1, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, 3)
            cropped = frame1[bottom_right_y:top_left_y, top_left_x:bottom_right_x]
            #frame1 = cv2.flip(frame1, 1)
            matches = sift_detector(cropped, image_template)
            cv2.putText(frame1, str(matches), (450, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
            threshold = 4
            if matches > threshold:
                cv2.rectangle(frame1, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 3)
                cv2.putText(frame1, 'Каллибровка камеры прошла успешно', (5, 5), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                token += 1
                if token == 3:
                    sift = False
                    token = 0

        #----- Обновления фрейма и вывод на экран результата ----

        frame1 = cv2.resize(frame1, (780, 520))
        imgbytes = cv2.imencode('.png', frame1)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
        ret, frame1 = cap.read()

main()
