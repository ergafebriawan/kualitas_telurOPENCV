import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import imutils

cap = cv2.VideoCapture(0)
#(ret, frame) = cap.read()
colorGood = (255,255,255)
prediction = 'n.a.'

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

    # Capture frame-by-frame
    _,img = cap.read(0) #inisialisasi video
    img = imutils.resize(img, width=720) #mengatur ukuran video

    cv2.putText(
        img,
        'kualitas telur: ' + prediction,
        (15, 25),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        colorGood,
        )

    # Display the resulting frame
    cv2.imshow('color classifier', img)

    color_histogram_feature_extraction.color_histogram_of_test_image(img)

    prediction = knn_classifier.main('training.data', 'test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()		
