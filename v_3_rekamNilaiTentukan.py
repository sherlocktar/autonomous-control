import numpy as np
import cv2
#import time
import matplotlib.pyplot as plt
from PIL import ImageGrab

#number = []
#kecepatan = None
def kecepatan(image):
    image = image[524:560,687:754]
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
    #-------------------------------_ _ c

    a,b,c,d,e,f,g = 0,0,0,0,0,0,0
    number_c = []
    if(thresh[8,58] > 0):
        a = 1
    else:
        a = 0

    if(thresh[13,64] > 0):
        b = 1
    else:
        b = 0
        
    if(thresh[23,61] > 0):
        c = 1
    else:
        c = 0

    if(thresh[29,54] > 0):
        d = 1
    else:
        d = 0

    if(thresh[23,48] > 0):
        e = 1
    else:
        e = 0

    if(thresh[13,51] > 0):
        f = 1
    else:
        f = 0

    if(thresh[18,56] > 0):
        g = 1
    else:
        g = 0
    number_c = [a,b,c,d,e,f,g]
    if number_c == [1,1,1,1,1,1,0]:
        #print("1")
        kecepatan_c = '0'
    elif number_c == [0,1,1,0,0,0,0]:
        #print("1")
        kecepatan_c = '1'
    elif number_c == [1,1,0,1,1,0,1]:
        #print("2")
        kecepatan_c = '2'
    elif number_c == [1,1,1,1,0,0,1]:
        #print("3")
        kecepatan_c = '3'
    elif number_c == [0,1,1,0,0,1,1]:
        #print("4")
        kecepatan_c = '4'
    elif number_c == [1,0,1,1,0,1,1]:
        #print("5")
        kecepatan_c = '5'
    elif number_c == [1,0,1,1,1,1,1]:
        #print("6")
        kecepatan_c = '6'
    elif number_c == [1,1,1,0,0,0,0]:
        #print("7")
        kecepatan_c = '7'
    elif number_c == [1,1,1,1,1,1,1]:
        #print("8")
        kecepatan_c = '8'
    elif number_c == [1,1,1,1,0,1,1]:
        #print("9")
        kecepatan_c = '9'
    #--------------------------------_ b _

    h,i,j,k,l,m,n = 0,0,0,0,0,0,0
    number_b = []
    if(thresh[8,38] > 0):
        h = 1
    else:
        h = 0
    if(thresh[13,43] > 0):
        i = 1
    else:
        i = 0
        
    if(thresh[23,40] > 0):
        j = 1
    else:
        j = 0

    if(thresh[29,33] > 0):
        k = 1
    else:
        k = 0

    if(thresh[23,27] > 0):
        l = 1
    else:
        l = 0

    if(thresh[13,29] > 0):
        m = 1
    else:
        m = 0

    if(thresh[18,35] > 0):
        n = 1
    else:
        n = 0
    number_b = [h,i,j,k,l,m,n]
    if number_b == [0,0,0,0,0,0,0]:
        kecepatan_b = ''
    else:
        if number_b == [0,1,1,0,0,0,0]:
            #print("1")
            kecepatan_b = '1'
        elif number_b == [1,1,0,1,1,0,1]:
            #print("2")
            kecepatan_b = '2'
        elif number_b == [1,1,1,1,0,0,1]:
            #print("3")
            kecepatan_b = '3'
        elif number_b == [0,1,1,0,0,1,1]:
            #print("4")
            kecepatan_b = '4'
        elif number_b == [1,0,1,1,0,1,1]:
            #print("5")
            kecepatan_b = '5'
        elif number_b == [1,0,1,1,1,1,1]:
            #print("6")
            kecepatan_b = '6'
        elif number_b == [1,1,1,0,0,0,0]:
            #print("7")
            kecepatan_b = '7'
        elif number_b == [1,1,1,1,1,1,1]:
            #print("8")
            kecepatan_b = '8'
        elif number_b == [1,1,1,1,0,1,1]:
            #print("9")
            kecepatan_b = '9'    
    try:
        kecepatan = int(kecepatan_b + kecepatan_c)
    except Exception as e:
        print(e)
        kecepatan = -1
        return kecepatan
    return kecepatan

'''
for i in range(4)[::-1]:
        print(i+1)
        time.sleep(1)

pos_1 = 0
n = 1
while(True):
        #ambil layar
    image = np.array(ImageGrab.grab(bbox=(5,40,800,625)))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #ubah bgr ke rgb
    kec_mob = kecepatan(image)
    
    #plt.imshow(thresh)
    #plt.show()
    print(kec_mob)
    #plt.imshow(image)
    #plt.show()

    if cv2.waitKey(25) & 0xFF == ord('q'):
            #cv2.imwrite('u1.jpg',gambar_warped)
        cv2.destroyAllWindows()
        break


'''
    
