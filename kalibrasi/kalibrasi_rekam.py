import numpy as np
from PIL import ImageGrab
import cv2
import time
import matplotlib.pyplot as plt
import serial
import time
np.warnings.filterwarnings('ignore')
def nothing(x):
    pass

def maju():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.1)
    ReleaseKey(W)
    
def kekiri():
    #maju()
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(S)
    time.sleep(0.1)
    ReleaseKey(A)

def kekanan():
    #maju()
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    time.sleep(0.1)
    ReleaseKey(D)
def slow():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    time.sleep(0.1)
def rem():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(0.1)
    
#ubah pixel
def rescaleFrame(frame, scale):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv2.resize(frame, dimensions, interpolation= cv2.INTER_AREA)


#gambar poligon
def dap_ROI(img , vertices):
    if len(img.shape) == 3:
        fill_color = (255,) * 3
    else:
        fill_color = 255
   # vertices = np.array(vertices, ndmin=3, dtype=np.int32)
    mask = np.zeros_like(img)
    #cv2.imshow('mask ini',mask)
    mask = cv2.fillPoly(mask, vertices, (255,255,255))
    masked_or = cv2.bitwise_or(img, mask)
    masked_and = cv2.bitwise_and(img, mask)
    return masked_or , masked_and

#buat melengkung yang dibatasi
def proses_warp(img, warp_shape , src, dst):
    M = cv2.getPerspectiveTransform(src,dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M , warp_shape, flags= cv2.INTER_LINEAR)
   #warped = warped.transpose( (2,0,1) )
    #cv2.imshow('hasil',warped)
    return warped, M , invM
def kalibrasi_thres(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    r = cv2.getTrackbarPos('thresh','image')
    ret,thresh = cv2.threshold(gray,193,255,cv2.THRESH_BINARY)
    return thresh

def process_img(image):
    img_rescal = image
    ysize = img_rescal.shape[0]
    xsize = img_rescal.shape[1]
    # 1. Mengambil kordinat ROI
    #kordinat_roi = np.array([ [0,392] , [320,280] ,[480,280] ,[800,392] ])
    kordinat_roi = np.array([ [80,392] , [260,280] ,[540,280] ,[720-1,392] ])
    gambar_roi_or , gambar_roi_and  = dap_ROI(img_rescal , [kordinat_roi])

    # 2. Mengambil tampilan burung
    src = np.float32([ [260,280] ,
                       [80,392] ,
                       [720-1,392] ,
                       [ 540,280] ])
    dst = np.float32([[0 ,0],[100, ysize],[xsize-101, ysize],[xsize-1, 0]])
    #print(dst)


    
    kordinat_warp = np.array([[0 ,0],[0, ysize],[xsize, ysize],[xsize, 0]])
    gambar_warped, M , invM = proses_warp(gambar_roi_and , (xsize,ysize) ,src,dst)
    
    #pts = pts.reshape( (-1,1,2) ) #tergrantung permintaan
    cv2.polylines(gambar_roi_and, [kordinat_roi] , True, (0, 255, 0), 1)
    cv2.polylines(img_rescal, [kordinat_roi] , True, (0, 255, 0), 1)
        
    return gambar_roi_and,gambar_warped,img_rescal

def graynya_beb(img2,n_gr):
    gray2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    ar_gr = n_gr
    ret, ga_put_gr = cv2.threshold(gray2,ar_gr,255,cv2.THRESH_BINARY)
    return ga_put_gr


def hsvnya_beb(screen,s_ku,v_pu):
    hsv = cv2.cvtColor(screen,cv2.COLOR_RGB2HSV)
    hue_hsv = hsv[: , : ,0]
    sat_hsv = hsv[: , : ,1]
    val_hsv = hsv[: , : ,2]

    #===================(KUNING)================
    cjb_hsv_ku = s_ku#90 #103
    ga_hsv_ku = cv2.inRange(hsv,(10,cjb_hsv_ku,0),(42,255,255))
    #cv2.imshow('garis kuning hsv',ga_hsv_ku) 
    #=============================================

    #==================(PUTIH)====================
    r_hsv_pu = v_pu #191
    ga_hsv_pu = cv2.inRange(hsv,(0,0,r_hsv_pu),(255,255,255))
    #cv2.imshow('garis putih hsv',ga_hsv_pu)
    #================================================
    #+++++++++++++++++++++++++Tampilkan hasil HSV+++++++++
    gabung_hsv = cv2.bitwise_or(ga_hsv_ku,ga_hsv_pu)
    return gabung_hsv

    
def main():
    cv2.namedWindow('image')
    cv2.createTrackbar('S_Ku','image',0,255,nothing)
    cv2.createTrackbar('V_Pu','image',0,255,nothing)
    cv2.createTrackbar('N_Gr','image',0,255,nothing)
    for i in range(4)[::-1]:
        print(i+1)
        time.sleep(1)

    pos_1 = 0
    n = 1
    while(True):
        #ambil layar
        screen = np.array(ImageGrab.grab(bbox=(5,40,800,625)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB) #ubah bgr ke rgb
        gambar_roi_and, gambar_warped, img_rescal = process_img(screen)
        
        s_ku = cv2.getTrackbarPos('S_Ku','image')
        v_pu = cv2.getTrackbarPos('V_Pu','image')
        n_gr = cv2.getTrackbarPos('N_Gr','image')

        gab_hsvnya = hsvnya_beb(gambar_warped,s_ku,v_pu)
        graynya = graynya_beb(gambar_warped,n_gr)
        
        kombinasi = np.array(graynya + gab_hsvnya)
        kombinasi[kombinasi == (253 or 254)] = 255

        #kuning-putih
        
        kal = rescaleFrame(kombinasi, 0.45)
        cv2.imshow('image warp - kal',kal) #ubah ini
        
        #putih-putih
        '''
        kal = rescaleFrame(graynya, 0.45)
        cv2.imshow('gary',kal) #ubah ini
        '''
        
        

        #gambar_warped= rescaleFrame(gambar_warped, 0.45)#
        #out = rescaleFrame(owawwwut, 0.45)        
        
        #cv2.imshow('image roi - and',gambar_roi_and) #ubah ini
        #cv2.imshow('image warp',gambar_warped) #ambil ini
        #cv2.imshow('image rescal',img_rescal)
        #cv2.imshow('hasil sliding',out)
       
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            #cv2.imwrite('u1.jpg',gambar_warped)
            cv2.destroyAllWindows()
            break

main()
