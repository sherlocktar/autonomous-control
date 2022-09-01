#---------------------------Program Kemudi Otonom------------------
#----Nama : Daud Barita Lambok Tarigan
#----NIM  : 170402095
#----Dosen Pembimbing : Suherman,ST,M.Comp,Ph.D
#----Dosen Pembanding 1 : Emerson P Sinulingga,ST,M.Sc,Ph.D
#----Dosen Pembanding 2 : Soeharwinto,ST,MT
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import serial
import time
import hid
from PIL import ImageGrab
from v2_kombinasi_gray_hsv_fungsi import binerisasi
from v_2_detekGaris_fungsi import detectGaris
from v3_direct_keys import PressKey,ReleaseKey,W,A,S,D
from v_3_rekamNilaiTentukan import kecepatan
import pandas as pd

np.warnings.filterwarnings('ignore')
arduino = serial.Serial('COM11', 9600)

#------------baca sudut kemudi
vid = 0x10c4	# Change it for your device
pid = 0x82c0	# Change it for your device

col1 = "Sudut_P"
list1 = []
col2 = "Sudut_A"
list2 = []
col3 = "Waktu"
list3 = []
col4 = "Waktu_Pro"
list4 = []
col5 = "Cycle"
list5 = []

col6 = "waktu pos"
list6 = []
#-------
daud_kec = 10

#-------------------fungsi Keyboard---------------
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
#---------------------------------------------------------------


#-------------------fungsi untuk rescaling gambar---------------
def rescaleFrame(frame, scale):
    # Images, Videos dan Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation= cv2.INTER_AREA)


#-------------------Dapatkan ROI-------------------
def dap_ROI(img , vertices):
    if len(img.shape) == 3:
        fill_color = (255,) * 3
    else:
        fill_color = 255
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, vertices, (255,255,255))
    masked_or = cv2.bitwise_or(img, mask)
    masked_and = cv2.bitwise_and(img, mask)
    return masked_or , masked_and

#-------------------Bird's Eye View Transformation-------------------
def proses_warp(img, warp_shape , src, dst):
    M = cv2.getPerspectiveTransform(src,dst)
    invM = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M , warp_shape, flags= cv2.INTER_LINEAR)
    return warped, M , invM

#-------------------binerisasi warna putih-------------------
def kalibrasi_thres(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    r = cv2.getTrackbarPos('thresh','image')
    ret,thresh = cv2.threshold(gray,163,255,cv2.THRESH_BINARY)
    return thresh

#-------------------Bird's Eye View-------------------
def process_img(image):
    img_rescal = image
    ysize = img_rescal.shape[0]
    xsize = img_rescal.shape[1]
    kordinat_roi = np.array([ [80,392] , [260,280] ,[540,280] ,[720-1,392] ])
    gambar_roi_or , gambar_roi_and  = dap_ROI(img_rescal , [kordinat_roi])
    src = np.float32([ [260,280] ,
                       [80,392] ,
                       [720-1,392] ,
                       [ 540,280] ])
    dst = np.float32([[0 ,0],[100, ysize],[xsize-101, ysize],[xsize-1, 0]])
    kordinat_warp = np.array([[0 ,0],[0, ysize],[xsize, ysize],[xsize, 0]])
    gambar_warped, M , invM = proses_warp(gambar_roi_and , (xsize,ysize) ,src,dst)
    cv2.polylines(gambar_roi_and, [kordinat_roi] , True, (0, 255, 0), 1)
    cv2.polylines(img_rescal, [kordinat_roi] , True, (0, 255, 0), 1)
        
    return gambar_roi_and,gambar_warped,img_rescal

def inti(screen):
    gambar_roi_and, gambar_warped, img_rescal = process_img(screen)
    #--------------kalibrasi
    '''Jika marka kiri dan kanan putih maka pilih:
            kal = kalibrasi_thres(gambar_warped)
        Jika marka kiri dan kanan putih atau kuning maka pilih:
            kal =binerisasi(gambar_warped)
        Note : komentarkan salah satu
    '''
    kal =binerisasi(gambar_warped)       #putih kuning
    #kal = kalibrasi_thres(gambar_warped)  #putih
    #==============================================
    out,offset = detectGaris(kal)
    kec_mob = kecepatan(screen)
    #--------------pilih kontrol
    gambar_roi_and = rescaleFrame(gambar_roi_and,0.45)
    img_rescal = rescaleFrame(img_rescal, 0.45)
    kal = rescaleFrame(kal, 0.45)

    gambar_warped= rescaleFrame(gambar_warped, 0.45)
    out = rescaleFrame(out, 0.45)
        
    cv2.imshow('image warp - kal',kal) 
    cv2.imshow('hasil sliding',out)

    return offset,kec_mob


def kon_tanpa_Detek_Dan_Manual(screen):
    gambar_manual= rescaleFrame(screen, 0.45)
    cv2.imshow('gambar manual',gambar_manual)
    kec_mob = kecepatan(screen)
    return kec_mob
    
    

def kon_Detek_Dan_kemudi(screen):
    offset,kec_mob = inti(screen)
    ####-------kiri
    #'''
    if offset < -175:
        #sud_per = 180
        err_per = -14
        arduino.write('o'.encode())
        print('kekiri1')
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -162:
        #kekanan()
        #sud_per = 174
        err_per = -13
        print('kekiri2')
        arduino.write('n'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -149:
        #kekanan()
        #sud_per = 168
        err_per = -12
        print('kekiri3')
        arduino.write('m'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -136:
        #kekanan()
        #sud_per = 162
        err_per = -11
        print('kekiri4')
        arduino.write('l'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -123:
        #kekanan()
        #sud_per = 156
        err_per = -10
        print('kekiri5')
        arduino.write('k'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -110:
        #kekanan()
        #sud_per = 150
        err_per = -9
        print('kekiri6')
        arduino.write('j'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -97:
        #kekanan()
        #sud_per = 144
        err_per = -8
        print('kekiri7')
        arduino.write('i'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -84:
        #kekanan()
        #sud_per = 138
        err_per = -7
        print('kekiri8')
        arduino.write('h'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -71:
        #kekanan()
        #sud_per = 132
        err_per = -6
        print('kekiri9')
        arduino.write('g'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -58:
        #kekanan()
        #sud_per = 126
        err_per = -5
        print('kekiri10')
        arduino.write('f'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -45:
        #kekanan()
        #sud_per = 120
        err_per = -4
        print('kekiri11')
        arduino.write('e'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -32:
        #kekanan()
        #sud_per = 114
        err_per = -3
        print('kekiri11')
        arduino.write('d'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -19:
        #kekanan()
        #sud_per = 108
        err_per = -2
        print('kekiri12')
        arduino.write('c'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset < -6:
        #kekanan()
        #sud_per = 102
        err_per = -1
        print('kekiri13')
        arduino.write('b'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    ####--------kanan
    elif offset > 175:
        #sud_per = 0
        print('kekanan1')
        err_per = 14
        arduino.write('O'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 162:
        #sud_per = 6
        err_per = 13
        print('kekanan2')
        arduino.write('N'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 149:
        #sud_per = 12
        err_per = 12
        print('kekanan3')
        arduino.write('M'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 136:
        #sud_per = 18
        err_per = 11
        print('kekanan4')
        arduino.write('L'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 123:
        #sud_per = 24
        err_per = 10
        print('kekanan5')
        arduino.write('K'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 110:
        #sud_per = 30
        err_per = 9
        print('kekanan6')
        arduino.write('J'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 97:
        #sud_per = 36
        err_per = 8
        print('kekanan7')
        arduino.write('I'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 84:
        #sud_per = 42
        err_per = 7
        print('kekanan8')
        arduino.write('H'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 71:
        #sud_per = 48
        err_per = 6
        print('kekanan9')
        arduino.write('G'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 58:
        #sud_per = 54
        err_per = 5
        print('kekanan10')
        arduino.write('F'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 45:
        #sud_per = 60
        err_per = 4
        print('kekanan11')
        arduino.write('E'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 32:
        #sud_per = 66
        err_per = 3
        print('kekanan12')
        arduino.write('D'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 19:
        #sud_per = 72
        err_per = 2
        print('kekanan13')
        arduino.write('C'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    elif offset > 6:
        #sud_per = 78
        err_per = 1
        print('kekanan14')
        arduino.write('B'.encode())
        if kec_mob < daud_kec:
            maju()
        else:
            slow()
    else :
        #####maju   -6 dan 6
        #sud_per = 90
        err_per = 0
        print('maju')
        arduino.write('a'.encode())
        if kec_mob > daud_kec:
            slow()
            print('perlambat lu')
        else:
            maju()
            print('maju cuk')
    #'''
    return offset,kec_mob #,err_per 

def Rekam_Pengujian_fps():
    global prev_frame_time
    global new_frame_time
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    return fps

def peta_var(nilai):
    in_min = 76
    in_max = 179
    out_min = 0
    out_max = 180
    hasil = ((nilai-in_min) * (out_max-out_min)/(in_max - in_min)) + out_min
    return hasil

def baca_kemudi():
    report = gamepad.read(64)
    if report:
        #print(report[3])
        nids = int(peta_var(report[3]))
        if nids < (0 + 3):
            err_sud = 14
        elif nids < (6 + 3):
            err_sud = 13
        elif nids < (12 + 3):
            err_sud = 12
        elif nids < (18 + 3):
            err_sud = 11
        elif nids < (24 + 3):
            err_sud = 10
        elif nids < (30 + 3):
            err_sud = 9
        elif nids < (36 + 3):
            err_sud = 8
        elif nids < (42 + 3):
            err_sud = 7
        elif nids < (48 + 3):
            err_sud = 6
        elif nids < (54 + 3):
            err_sud = 5
        elif nids < (60 + 3):
            err_sud = 4
        elif nids < (66 + 3):
            err_sud = 3
        elif nids < (72 + 3):
            err_sud = 2
        elif nids < (78 + 3):
            err_sud = 1
        elif nids < (90 + 3):
            err_sud = 0
        elif nids < (102 + 3):
            err_sud = -1
        elif nids < (108 + 3):
            err_sud = -2
        elif nids < (114 + 3):
            err_sud = -3
        elif nids < (120 + 3):
            err_sud = -4
        elif nids < (126 + 3):
            err_sud = -5
        elif nids < (132 + 3):
            err_sud = -6
        elif nids < (138 + 3):
            err_sud = -7
        elif nids < (144 + 3):
            err_sud = -8
        elif nids < (150 + 3):
            err_sud = -9
        elif nids < (156 + 3):
            err_sud = -10
        elif nids < (162 + 3):
            err_sud = -11
        elif nids < (168 + 3):
            err_sud = -12
        elif nids < (174 + 3):
            err_sud = -13
        else:
            err_sud = -14

    return err_sud
        

def main():
    for i in range(4)[::-1]:
        print(i+1)
        time.sleep(1)

    pos_1 = 0
    n = 1
    global prev_frame_time
    global new_frame_time

    prev_frame_time = 0
    new_frame_time = 0
    waktuSeimbang = 0
    mulai = time.time()

    global gamepad
    gamepad = hid.device()
    gamepad.open(vid, pid)
    gamepad.set_nonblocking(True)

    cikel = 0
    y = 0
    sek=0
    while(True):
        
        screen = np.array(ImageGrab.grab(bbox=(5,40,800,625)))
        screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
        #--------------Pilihs kontrol-----------------
        #kec_mob = kon_tanpa_Detek_Dan_Manual(screen)
        offset,kec_mob  = kon_Detek_Dan_kemudi(screen)
        #=====baca sudut kemudi
        #err_sud = baca_kemudi()  #2
        
        #============================================        
        #---------------Pilih Pengujian--------------
        #=Pengujian FPS------------------------------
        #fps = Rekam_Pengujian_fps()
        #list1.append(int(fps))
        #============================================
        #=Pengujian error----------------------------
        #error = abs(offset)
        #list1.append(error)
        #============================================
        #=Pengujian waktu posisi kiri dan kanan------
        #print(offset)
        
        if sek == 0:
            if offset > -6 and offset < 6:
                akhir = time.time()
                waktuSeimbang = akhir - mulai
                mulai = akhir
                waktu_pos = waktuSeimbang
                list1.append(waktu_pos)
                sek = 1
                print("dah masuk")
        
        #==========================================
        #list1.append(waktu_pro)
        #list2.append(cikel)        
        #list1.append(err_per)
        #list2.append(err_sud)
        #list3.append(waktunya)
        #list4.append(waktu_pro)
        #list5.append(cikel)
        #==aw==/*========================================
        #cv2.imshow('gdwambar manual',screen)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            #cv2.imwrite('jalan1.jpg',screen)
            #----fps dan error-------------------
            data = pd.DataFrame({col1:list1})
            #----waktu posisi-------------------
            #data = pd.DataFrame({col1:list1,col2:list2,col3:list3,col4:list4,col5:list5 })
            #----sudut perintah dan aktual-------------------
            #data = pd.DataFrame({col1:list1,col2:list2 })
            #++++cetak ke excel++++
            #data.to_excel('kemudi_sudut_p_dan_aktual_p3.xlsx', sheet_name='sheet1', index=False)
            data.to_excel('waktu_posisi_p3_ka_kec.xlsx', sheet_name='sheet1', index=False)
            cv2.destroyAllWindows()
            break

main()


