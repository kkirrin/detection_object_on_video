import sys
import cv2
import numpy as np
import time


def main():
    
  # Передаем через консоль первый агрумент (название видео)
  file_name = sys.argv[1]

  video = cv2.VideoCapture(file_name)

  # формируем начальный и конечный цвет фильтра
  h_min = np.array((11, 0, 0), np.uint8)
  h_max = np.array((245, 245, 250), np.uint8)

  # Делаем покадровый захват
  while(video.isOpened()):
      
    ret, frame = video.read()

    # Сделаем бинаризацию
    # frame_2 = cv2.bilateralFilter(frame, 9,75,75)

    hsv = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
    # Накладываем фильтр на кадр в модели HSV
    thresh = cv2.inRange(hsv, h_min, h_max)

    # Также погасим шумы
    blur = cv2.GaussianBlur(thresh, (3,3), 0)

    # ищем контуры и складируем их в переменную contours
    contours, hierarchy = cv2.findContours(blur.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(blur, contours,-1,(0,255,0),3, cv2.LINE_AA, hierarchy, 1) 


    # Отображение конечного результата
    cv2.imshow('frame', blur)
    
    # Если закончилось и нажата клавиша 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


  # Определим, за сколько прошла обработка и сохраним в переменной
  begtime = time.perf_counter()
  endtime = time.perf_counter()
  amount_of_seconds = {endtime - begtime}
  print(f"\n Затрачено, с: {amount_of_seconds} ")  


  video.release()
  cv2.destroyAllWindow()


if __name__ == "__main__":
  main()
