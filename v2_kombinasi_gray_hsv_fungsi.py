import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



def binerisasi(img2):
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    #--------------------gray--------------------------------------------wwwwww------
    #==========(putih)===================
    gray2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    ar_gr = 153
    #print(ar)
    #+++++++++++++++++++tampilkan hasil gray++++++++++
    ret,ga_put_gr = cv2.threshold(gray2,ar_gr,255,cv2.THRESH_BINARY)
    #cv2.imshow('garis putih gray',ga_put_gr)
    #-----------------------------------------------------------------------
    #--------------------HSV-----------------------------------------------
    hsv = cv2.cvtColor(img2,cv2.COLOR_RGB2HSV)
    hue_hsv = hsv[: , : ,0]
    sat_hsv = hsv[: , : ,1]
    val_hsv = hsv[: , : ,2]
    #===================(KUNING)================
    cjb_hsv_ku = 102
    ga_hsv_ku = cv2.inRange(hsv,(10,cjb_hsv_ku,0),(42,255,255))
    #cv2.imshow('garis kuning hsv',ga_hsv_ku) 
    #=============================================
    #==================(PUTIH)====================
    r_hsv_pu = 153
    ga_hsv_pu = cv2.inRange(hsv,(0,0,r_hsv_pu),(255,255,255))
    #cv2.imshow('garis putih hsv',ga_hsv_pu)
    #================================================
    #+++++++++++++++++++++++++Tampilkan hasil HSV+++++++++
    gabung_hsv = cv2.bitwise_or(ga_hsv_ku,ga_hsv_pu)
    #cv2.imshow('gabung hsv',gabung_hsv)
    #------------------------------------------------------------------------
    #~~~~~~~~~~~~~~~~~~~~~gabung semua dengan syarat~~~~~~~~~~~~~~~~~~~~~~~~
    kombinasi = np.array(ga_put_gr + gabung_hsv)

    kombinasi[kombinasi == (253 or 254)] = 255
    return kombinasi




