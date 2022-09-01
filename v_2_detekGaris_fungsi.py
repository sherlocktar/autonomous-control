import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_poly_points(left_fit, right_fit):
    '''
    Get the points for the left lane/ right lane defined by the polynomial coeff's 'left_fit'
    and 'right_fit'
    :param left_fit (ndarray): Coefficients for the polynomial that defines the left lane line
    :param right_fit (ndarray): Coefficients for the polynomial that defines the right lane line
    : return (Tuple(ndarray, ndarray, ndarray, ndarray)): x-y coordinates for the left and right lane lines
    '''
    IMG_SHAPE = (800, 600)
    #print(left_fit)
    #print(right_fit)
    xsize,ysize = IMG_SHAPE
    #print(xsize-1)
    # Get the points for the entire height of the image
    plot_y = np.linspace(0, ysize-1, ysize)
    try:
        plot_xleft = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        plot_xright = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

    #xsize-1 -> kurangi data nol
    #print(len(plot_xleft)) 
    #print(len(plot_xright))

    # But keep only those points that lie within the image
        plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= xsize - 1)]
        plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= xsize - 1)]
        plot_yleft = np.linspace(ysize - len(plot_xleft), ysize - 1, len(plot_xleft))
        plot_yright = np.linspace(ysize - len(plot_xright), ysize - 1, len(plot_xright))
    except Exception as e:
        print(str(e))
        return False
    
    return plot_xleft.astype(int), plot_yleft.astype(int), plot_xright.astype(int), plot_yright.astype(int)



def check_validity(left_fit, right_fit, diagnostics=False):
    '''
    Determine the validity of lane lines represented by a set of second order polynomial coefficients 
    :param left_fit (ndarray): Coefficients for the 2nd order polynomial that defines the left lane line
    :param right_fit (ndarray): Coefficients for the 2nd order polynomial that defines the right lane line
    :param diagnostics (boolean): Boolean flag for logging
    : return (boolean)
    '''
    IMG_SHAPE = (800, 600)
    
    if left_fit is None or right_fit is None:
        return False
    
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(left_fit, right_fit)

    # Check whether the two lines lie within a plausible distance from one another for three distinct y-values

    y1 = IMG_SHAPE[0] - 1 # Bottom
    y2 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.35) # For the 2nd and 3rd, take values between y1 and the top-most available value.
    y3 = IMG_SHAPE[0] - int(min(len(plot_yleft), len(plot_yright)) * 0.75)

    # Compute the respective x-values for both lines
    x1l = left_fit[0]  * (y1**2) + left_fit[1]  * y1 + left_fit[2]
    x2l = left_fit[0]  * (y2**2) + left_fit[1]  * y2 + left_fit[2]
    x3l = left_fit[0]  * (y3**2) + left_fit[1]  * y3 + left_fit[2]

    x1r = right_fit[0] * (y1**2) + right_fit[1] * y1 + right_fit[2]
    x2r = right_fit[0] * (y2**2) + right_fit[1] * y2 + right_fit[2]
    x3r = right_fit[0] * (y3**2) + right_fit[1] * y3 + right_fit[2]

    # Compute the L1 norms
    x1_diff = abs(x1l - x1r)
    x2_diff = abs(x2l - x2r)
    x3_diff = abs(x3l - x3r)

    # Define the threshold values for each of the three points
    min_dist_y1 = 480 # 510 # 530 
    max_dist_y1 = 730 # 750 # 660
    min_dist_y2 = 280
    max_dist_y2 = 730 # 660
    min_dist_y3 = 140
    max_dist_y3 = 730 # 660
    
    if (x1_diff < min_dist_y1) | (x1_diff > max_dist_y1) | \
        (x2_diff < min_dist_y2) | (x2_diff > max_dist_y2) | \
        (x3_diff < min_dist_y3) | (x3_diff > max_dist_y3):
        if diagnostics:
            print("Violated distance criterion: " +
                  "x1_diff == {:.2f}, x2_diff == {:.2f}, x3_diff == {:.2f}".format(x1_diff, x2_diff, x3_diff))
        return False

    # Check whether the line slopes are similar for two distinct y-values
    # x = Ay**2 + By + C
    # dx/dy = 2Ay + B
    
    y1left_dx  = 2 * left_fit[0]  * y1 + left_fit[1]
    y3left_dx  = 2 * left_fit[0]  * y3 + left_fit[1]
    y1right_dx = 2 * right_fit[0] * y1 + right_fit[1]
    y3right_dx = 2 * right_fit[0] * y3 + right_fit[1]

    # Compute the L1-norm
    norm1 = abs(y1left_dx - y1right_dx)
    norm2 = abs(y3left_dx - y3right_dx)
    
#     if diagnostics: print( norm1, norm2)

    # Define the L1 norm threshold
    thresh = 0.6 #0.58 
    if (norm1 >= thresh) | (norm2 >= thresh):
        if diagnostics:
            print("Violated tangent criterion: " +
                  "norm1 == {:.3f}, norm2 == {:.3f} (thresh == {}).".format(norm1, norm2, thresh))
            return False
    
    return True

def plot_images(data, layout='row', cols=2, figsize=(20, 12)):
    '''
    Utility function for plotting images
    :param data [(ndarray, string)]: List of data to display, [(image, title)]
    :param layout (string): Layout, row-wise or column-wise
    :param cols (number): Number of columns per row
    :param figsize (number, number): Tuple indicating figure size
    '''
    rows = math.ceil(len(data) / cols)
    #print(data)
    f, ax = plt.subplots(figsize=figsize)
    #print(ax)
    #print(f)
    if layout == 'row':
        for idx, d in enumerate(data):
            img, title = d
            #print(idx)

            plt.subplot(rows, cols, idx+1)
            plt.title(title, fontsize=20)
            plt.axis('off')
            #print(len(img.shape))
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
                #plt.show()
                
            elif len(img.shape) == 3:
                plt.imshow(img)
                #plt.show()
                
    elif layout == 'col':
        counter = 0
        for r in range(rows):
            for c in range(cols):
                img, title = data[r + rows*c]
                nb_channels = len(img.shape)
                
                plt.subplot(rows, cols, counter+1)
                plt.title(title, fontsize=20)
                plt.axis('off')
                if len(img.shape) == 2:
                    plt.imshow(img, cmap='gray')
                
                elif len(img.shape) == 3:
                    plt.imshow(img)
                    #plt.show()
              
                counter += 1
    
    return ax

def compute_offset_from_center(poly_param):
    '''
    Computes the offset of the car from the center of the detected lane lines
    :param poly_param (ndarray): Set of 2nd order polynomial coefficients that represent the detected lane lines
    :param x_mppx (float32): metres/pixel in the x-direction
    :return (float32): Offset 
    '''
    IMG_SHAPE = (800, 600)
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(poly_param[0], poly_param[1])
    #print(plot_xleft)
    #print(plot_xright)
    #print(plot_xright)
    #print(plot_xleft)
    lane_center = (plot_xright[-1] + plot_xleft[-1]) / 2
    car_center = IMG_SHAPE[0] / 2
    #print(lane_center)
    #print(car_center)
    offset = (lane_center - car_center)
    #print(lane_center)
    return offset

def detectGaris(thresh):
    visualise = True
    diagnostics=True
    IMG_SHAPE = (800, 600)
    thresh = cv2.resize(thresh, IMG_SHAPE, interpolation= cv2.INTER_AREA)

    global cache
    ret = True
    binary = thresh
    # Sanity check
    if binary.max() <= 0:
        print ('False, np.array([]), np.array([]), np.array([])')
        
    histogram = None

    cutoffs = [int(binary.shape[0] / 2), 0]

    for cutoff in cutoffs:
        histogram = np.sum(binary[cutoff:, :], axis=0) #jumlah histogram berdasarkan sumbu x
        #print(histogram)
        if histogram.max() > 0:
            break

    if histogram.max() == 0:
            print('Unable to detect lane lines in this frame. Trying another frame!')
            print(' False, np.array([]), np.array([])')


    #print(histogram.min())
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #print(histogram)
    '''

    if visualise:
        #plot_images([(binary, 'Binary')])
        #plt.show()
        plt.plot(histogram, 'm', linewidth=2.0)
        
        #plt.show()
        plt.plot((midpoint, midpoint), (0, IMG_SHAPE[1]), 'c')
        #plt.show()
        plt.plot((0, IMG_SHAPE[0]), (cutoff, cutoff), 'c')
        #plt.show()
    '''
    #------------------------------------------------------
    out = np.dstack((binary,binary,binary))
    #print(out[207,642])
    #print(out)

    nb_windows = 12 # number of sliding windows
    margin = 100 # width of the windows +/- margin pjg sumbu y
    minpix = 50 # min number of pixels needed to recenter the window jarak dari center awal ke center akhir
    window_height = int(IMG_SHAPE[1] / nb_windows) #menentukan banyak window
    min_lane_pts = 10  # min number of 'hot' pixels needed to fit a 2nd order polynomial as a 
                        # lane line 10 gak ada ngaruh ke sliding window


    nonzero = binary.nonzero()
    nonzerox = np.array(nonzero[1]) #banyak angka nonzero berdasarkan sb x
    nonzeroy = np.array(nonzero[0])
    #print(binary[100,0:800])
    #print(nonzerox)
    #print(nonzeroy)

    leftx_current = leftx_base   #yg arg di histogram
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []
    for window in range(nb_windows):
        # Identify window boundaries in x and y (and left and right)
        win_y_low = IMG_SHAPE[1] - (1 + window) * window_height
        win_y_high = IMG_SHAPE[1] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw windows for visualisation
        cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),\
                          (0, 255, 0), 2)
        cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high),\
                          (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                             & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                             & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) >  minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract pixel positions for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
        
    left_fit, right_fit = None, None #tipe data nonetype tidak ada apa-apa

    #print(len(leftx))
    # Sanity check; Fit a 2nd order polynomial for each lane line pixels
    if len(leftx) >= min_lane_pts and len(rightx) >= min_lane_pts:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Validate detected lane lines
    #valid = check_validity(left_fit, right_fit, diagnostics=diagnostics)
    '''
    if not valid:
            # If the detected lane lines are NOT valid:
            # 1. Compute the lane lines as an average of the previously detected lines
            # from the cache and flag this detection cycle as a failure by setting ret=False
            # 2. Else, if cache is empty, return 
            
        if len(cache) == 0:
            if diagnostics: print('WARNING: Unable to detect lane lines in this frame.')
            print('False, np.array([]), np.array([])')
            
        avg_params = np.mean(cache, axis=0)
        left_fit, right_fit = avg_params[0], avg_params[1]
        ret = False
    '''
    #valid = check_validity(left_fit, right_fit, diagnostics=diagnostics)
            
    plot_xleft, plot_yleft, plot_xright, plot_yright = get_poly_points(left_fit, right_fit)

    # Color the detected pixels for each lane line
    out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 10, 255]
    #print(plot_xleft)
    #print(plot_yleft)
    left_poly_pts = np.array([np.transpose(np.vstack([plot_xleft, plot_yleft]))])
    right_poly_pts = np.array([np.transpose(np.vstack([plot_xright, plot_yright]))])
    #print(left_poly_pts)
    #print(right_poly_pts)
    # Plot the fitted polynomial
    cv2.polylines(out, np.int32([left_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)
    cv2.polylines(out, np.int32([right_poly_pts]), isClosed=False, color=(200,255,155), thickness=4)

    #-------------
    poly_param = np.array([left_fit,right_fit])
    offset = compute_offset_from_center(poly_param)

    #if visualise:
        #plot_images([(thresh, 'Original'), (out, 'Out')], figsize=(30, 40))
        #plt.show()
    return out,offset
#cv2.imshow('img',img)
#cv2.imshow('img biner',thresh)
#cv2.imshow('img histo',histogram)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
