import cv2
import numpy as np
from matplotlib import pyplot as plt 
from math import isclose
import time



if __name__  == "__main__":
    
    cv2.namedWindow("Sparse Optical Flow", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Dense Optical Flow", cv2.WINDOW_NORMAL)
  
    # Using resizeWindow()
    cv2.resizeWindow("Sparse Optical Flow", 900, 600)
    cv2.resizeWindow("Dense Optical Flow", 900, 600)
    
    
    cap = cv2.VideoCapture('Robots.mp4')
    ret, frame = cap.read()
    frame = cv2.resize(frame,(900,600))
        
    prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define parameters for the Lucas-Kanade method
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(frame)
    mask_sparse = np.zeros_like(frame)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255
    
    p0 = cv2.goodFeaturesToTrack(prev_gray_frame, mask=None, maxCorners=100, qualityLevel=0.3,
                             minDistance=7, blockSize=7)
    
    
    fps = 30
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame,(900,600))
        next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_cpy = frame.copy()
        
        if ret is True:
            
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            # Sparse optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_frame, gray_frame, p0, None, **lk_params)
            # Select only the keypoints that have moved
            good_new = p1[st==1]
            good_old = p0[st==1]
            #feat1 = cv2.goodFeaturesToTrack(prev_gray_frame,maxCorners=10000,qualityLevel=0.2,minDistance=10)
            #feat2 , status, error = cv2.calcOpticalFlowPyrLK(prev_gray_frame,gray_frame,feat1,None)
            
            # Draw the keypoints that have moved on the mask
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                                
                mask_sparse = cv2.line(mask_sparse,(a,b), (c,d), (0,255,0), 2)
                mask_sparse = cv2.circle(mask_sparse,(a,b), 5, (0,255,0), -1)
            
            bgr_sparse_frame = cv2.add(frame_cpy, mask_sparse)
            
            
            
            # Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, gray_frame, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.5, 0)
            mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1])
            mask[...,0] = ang*180 / np.pi / 2
            mask[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr_dense_frame = cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)
            
            
            prev_gray_frame = gray_frame
            
            # for i in range(len(feat1)):
            #     f10=int(feat1[i][0][0])
            #     f11=int(feat1[i][0][1])
            #     f20=int(feat2[i][0][0]) 
            #     f21=int(feat2[i][0][1])
            
            # #print(f10,f20,f11,f21)
            # if not isclose(f10,f20,abs_tol=0.8) or not isclose(f11,f21,abs_tol=0.8):
            #     cv2.line(frame, (f10,f11), (f20, f21), (0, 255, 0), 2)
            #     cv2.circle(frame, (f10, f11), 5, (0, 255, 0), -1)

            cv2.imshow('Sparse Optical Flow', bgr_sparse_frame)
            cv2.imshow('Dense Optical Flow',bgr_dense_frame)
            elapsed_time = time.time() - start_time
            
             # Delay for the appropriate amount of time to maintain the desired frame rate
            delay_time = int((1.0 / fps - elapsed_time) * 1000)
            if delay_time < 1:
                delay_time = 1
            key = cv2.waitKey(delay_time)
            
            if key == ord('q'):
                break
            
            p0 = good_new.reshape(-1, 1, 2)
            
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
