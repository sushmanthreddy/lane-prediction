import cv2
import numpy as np
import pandas as pd


def perspective_transformer():
    global src, dst, M, Minv

    img_width = 1280
    img_height = 720

    bot_width = .76 # percent of bottom
    mid_width = .17 #.17
    height_pct = .66 #.66
    bottom_trim = .935
    src = np.float32([
        [img_width*(.5-mid_width/2), img_height*height_pct],
        [img_width*(.5+mid_width/2), img_height*height_pct],
        [img_width*(.5+bot_width/2), img_height*bottom_trim],
        [img_width*(.5-bot_width/2), img_height*bottom_trim]
    ])

    offset = img_width*.2

    dst = np.float32([
        [offset, 0],
        [img_width-offset, 0],
        [img_width-offset, img_height],
        [offset, img_height]
    ])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)



def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(sobelx**2+sobely**2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(gradient_direction)
    dir_binary[(gradient_direction>=thresh[0])&(gradient_direction<=thresh[1])] = 1
    return dir_binary

def color_thresh(img, s_thresh=(0,255), v_thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel>=s_thresh[0]) & (s_channel<=s_thresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel>=v_thresh[0]) & (v_channel<=v_thresh[1])] = 1

    c_binary = np.zeros_like(s_channel)
    c_binary[(s_binary==1) & (v_binary==1)] = 1

    return c_binary

def thresh_pipeline(img, gradx_thresh=(0,255), grady_thresh=(0,255), s_thresh=(0, 255), v_thresh=(0, 255)):
    gradx = abs_sobel_thresh(img, orient='x', thresh=gradx_thresh)
    grady = abs_sobel_thresh(img, orient='y', thresh=grady_thresh)
    c_binary = color_thresh(img, s_thresh=s_thresh, v_thresh=v_thresh)
    thresh_binary = np.zeros_like(img[:,:,0])
    thresh_binary[(gradx==1) & (grady==1) | (c_binary==1)] = 255
    return thresh_binary


def warper(img, src, dst):

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped



def sliding_windows_search(img):
    binary_warped = img.astype('uint8')
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    window_height = int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, out_img


def cal_curvature(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/505/1.2054/0.97 # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    lane_center = (left_fitx[-1]+right_fitx[-1])/2
    center_diff = (640-lane_center)*xm_per_pix
    return left_curverad, right_curverad, center_diff


def lane_mask(img_undist, binary_warped, Minv, ploty, left_fitx, right_fitx):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    middle_x = (left_fitx + right_fitx)/2
    middle_pts = np.transpose(np.vstack((middle_x, ploty))).astype(np.int32)
    cv2.polylines(color_warp, np.int32([pts_left]), False, (0, 0, 255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), thickness=15)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_undist.shape[1], img_undist.shape[0]))
    result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)
    return result


def lane_quality(ploty, left_fitx, right_fitx):
    xm_per_pix = 3.7/505/1.2054/0.97
    lane_width = (right_fitx - left_fitx)
    lane_width_mean = np.mean(lane_width)*xm_per_pix
    lane_width_var = np.var(lane_width)

    return lane_width_mean, lane_width_var


class Line():
    def __init__(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0,0,0], dtype='float')
        self.allx = None
        self.ally = None


def lane_tracking(img, left_fit, right_fit):
    binary_warped = img.astype('uint8')
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    window_img = np.zeros_like(out_img)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return left_fit, right_fit, result


def lane_finding(img_orig, line_l, line_r, mtx, dist):

    perspective_transformer()

    img_undist = cv2.undistort(img_orig, mtx, dist, None, mtx)

    img_thresh = thresh_pipeline(img_undist, gradx_thresh=(25,255), grady_thresh=(10,255), s_thresh=(100, 255), v_thresh=(0, 255))

    img_birdeye = warper(img_thresh, src, dst)
    img_birdeye_color = warper(img_undist, src, dst)

    if (not line_l.detected) or (not line_r.detected):
        left_fit, right_fit, img_search = sliding_windows_search(img_birdeye)


    else:
        left_fit, right_fit, img_search = lane_tracking(img_birdeye,
                                                        line_l.recent_xfitted[-1][0],
                                                        line_r.recent_xfitted[-1][0])

    line_l.current_fit = [left_fit]
    line_r.current_fit = [right_fit]

    line_l.bestx = None
    line_r.bestx = None

    line_l.recent_xfitted.append([left_fit])
    line_r.recent_xfitted.append([right_fit])

    if len(line_l.recent_xfitted)>1:

        line_l.best_fit = np.mean(np.array(line_l.recent_xfitted[-20:-1]),
                                axis=0)
        line_r.best_fit = np.mean(np.array(line_r.recent_xfitted[-20:-1]),
                                axis=0)
    else:
        line_l.best_fit = line_l.recent_xfitted[-1][0]
        line_r.best_fit = line_r.recent_xfitted[-1][0]

    ploty = np.linspace(0, img_birdeye.shape[0]-1, img_birdeye.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_curverad, right_curverad, center_diff = cal_curvature(ploty, left_fitx, right_fitx)
    lane_width_mean, lane_width_var = lane_quality(ploty, left_fitx, right_fitx)

    line_l.diffs = left_fit - line_l.best_fit

    line_r.diffs = right_fit - line_r.best_fit

    lane_continue = np.sum(line_l.diffs**2)+np.sum(line_r.diffs)

    if (not 3<lane_width_mean<5) or (lane_width_var>500) or (lane_continue>6000):

        line_l.detected = False
        line_r.detected = False

        del line_l.recent_xfitted[-1]

        del line_r.recent_xfitted[-1]

        left_fit, right_fit = line_l.best_fit[0], line_r.best_fit[0]

        cv2.putText(img_search, '------',
            (550, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)

        cv2.putText(img_search, 'Keeping',
            (550, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)

    else:
        line_l.detected = True
        line_r.detected = True

        line_l.best_fit = np.mean(np.array(line_l.recent_xfitted[-20:]),
                                axis=0)
        line_r.best_fit = np.mean(np.array(line_r.recent_xfitted[-20:]),
                                axis=0)


    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_curverad, right_curverad, center_diff = cal_curvature(ploty, left_fitx, right_fitx)
    lane_width_mean, lane_width_var = lane_quality(ploty, left_fitx, right_fitx)


    result = lane_mask(img_undist, img_birdeye, Minv, ploty, left_fitx, right_fitx)

    canvas = np.zeros([720,1280,3], dtype=np.uint8)

    canvas[0:720, 0:1280, :] = result


    return canvas
