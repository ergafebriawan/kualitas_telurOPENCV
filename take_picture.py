import cv2

cap=cv2.VideoCapture(0)
gambarr=1  # membuat variable untuk penamaan file yang akan ditulis nanti
top = 80
left = 100
bottom = 80
right = 100
while True :
    _,img=cap.read()
    cv2.imshow("tekan s untuk simpan foto",img)
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        cap.release()
        cv2.destroyAllWindows()
        break
    if k==ord('s'): # jika tombol 's' kecil ditekan
              h,w = img.shape[:2]
              # Menentukan Ukuran dan Resizing Image
              new_h, new_w = int(h/2),int(w/2)
              resizeImg = cv2.resize(img, (new_w,new_h))
              fileN=str(gambarr)+'.png' # membuat string nama image yang disimpan
              cv2.imwrite(fileN,resizeImg) # simpan image di folder yang aktif sekarang
              print (gambarr)
              gambarr = gambarr+1 # increase variable penamaan image agar penyimpanan selanjutnya tidak menimpa file yang lama