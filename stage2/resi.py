import cv2
im=cv2.imread('iclr+poster.png')
im=cv2.resize(im,(320,256))
cv2.imwrite('smallposter.png',im)