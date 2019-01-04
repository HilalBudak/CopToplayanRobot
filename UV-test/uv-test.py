import cv2
from datetime import datetime
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import colorsys
from PIL import Image
import _tkinter
import tkMessageBox


kamera=cv2.VideoCapture(0)

pencereIsmi="UV"
cv2.namedWindow(pencereIsmi)

t_eksi=cv2.cvtColor(kamera.read()[1],cv2.COLOR_BGR2GRAY)

zamanKontrol=datetime.now().strftime('%Ss')
i=0
while True:
    cv2.imshow(pencereIsmi,kamera.read()[1])
    i=i+1
    if i==1:
        fark_resim=kamera.read()[1]
	cv2.imwrite('q.jpg' ,fark_resim)
    key=cv2.waitKey(10)
    if key==27:
        cv2.destroyWindow(pencereIsmi)
        break

img_file = Image.open("q.jpg")
img = img_file.load()

# (2) Get image width & height in pixels
[xs, ys] = img_file.size
max_intensity = 100
hues = {}

# (3) Examine each pixel in the image file
for x in xrange(0, xs):
  for y in xrange(0, ys):
    # (4)  Get the RGB color of the pixel
    [r, g, b] = img[x, y]
    if r > 240 :
        if b>240:
            tkMessageBox.showinfo("Cop mu?", "COP")
            break


