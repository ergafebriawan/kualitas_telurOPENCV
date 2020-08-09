import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import imutils
from time import sleep
import RPi.GPIO as GPIO

cap = cv2.VideoCapture(0)
#(ret, frame) = cap.read()
colorGood = (255,255,255)
prediction = 'n.a.'

#pin use
led = 12 #definisi pin LED
switch = 11 #definisi pin limit switch
led_good = 29
led_bad = 36
buzzer = 37

#initial pin board
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led, GPIO.OUT)#inisialisasi pin LED
GPIO.setup(led_good, GPIO.OUT)
GPIO.setup(led_bad, GPIO.OUT)
GPIO.setup(buzzer, GPIO.OUT)

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('data training siap, memuat klasifikasi...')
else:
    print ('membuat data training...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('data training siap, memuat klasifikasi...')

while True:
	GPIO.output(led, 1)
	_,img = cap.read(0)
	img = imutils.resize(img, width=720)

	cv2.putText(img,'tekan D utk mendeteksi telur', (15, 25),cv2.FONT_HERSHEY_DUPLEX,0.7,colorGood)
	cv2.putText(img,'tekan Esc utk keluar program', (15, 50),cv2.FONT_HERSHEY_DUPLEX,0.7,colorGood)
	cv2.imshow('klasifikasi kualitas telur', img)
	

	k=cv2.waitKey(1) & 0xFF
	if k==27:
		cap.release()
		cv2.destroyAllWindows()
		break
	if k==ord('d'): 
		cv2.imwrite('telur.png',img)
		print("gambar disimpan")
		sleep(2)

		uncropimg = cv2.imread('telur.png')
		h,w = uncropimg.shape[:2]
		resizeImg = cv2.resize(uncropimg, (w,h))
		print(resizeImg.shape)
		cropimg = resizeImg[150:400, 250:460]

		color_histogram_feature_extraction.color_histogram_of_test_image(cropimg)
		prediction = knn_classifier.main('training.data', 'test.data')
		cv2.putText(cropimg,'kualitas telur:',(15, 25),cv2.FONT_HERSHEY_DUPLEX,0.5,colorGood)
		cv2.putText(cropimg,prediction,(15, 50),cv2.FONT_HERSHEY_DUPLEX,0.5,colorGood)
		cv2.imshow('cropTelur.png', cropimg)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()		
