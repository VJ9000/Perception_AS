import cv2
import numpy as np
from matplotlib import pyplot as plt 
from math import isclose




if __name__  == "__main__":
    
    cap = cv2.VideoCapture('Robots.mp4')
    ret, frame = cap.read()
        
    prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(frame)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255
    
    while (True):
        ret, frame = cap.read()
        next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        if ret is True:
            
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            # Sparse optical Flow
            feat1 = cv2.goodFeaturesToTrack(prev_gray_frame,maxCorners=10000,qualityLevel=0.2,minDistance=10)
            feat2 , status, error = cv2.calcOpticalFlowPyrLK(prev_gray_frame,gray_frame,feat1,None)
            
            
            # Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, gray_frame, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.5, 0)
            mag, ang = cv2.cartToPolar(flow[...,0],flow[...,1])
            mask[...,0] = ang*180 / np.pi / 2
            mask[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr_dense_frame = cv2.cvtColor(mask,cv2.COLOR_HSV2BGR)
            
            
            prev_gray_frame = gray_frame
            
            for i in range(len(feat1)):
                f10=int(feat1[i][0][0])
                f11=int(feat1[i][0][1])
                f20=int(feat2[i][0][0]) 
                f21=int(feat2[i][0][1])
            
            #print(f10,f20,f11,f21)
            if not isclose(f10,f20,abs_tol=0.8) or not isclose(f11,f21,abs_tol=0.8):
                cv2.line(frame, (f10,f11), (f20, f21), (0, 255, 0), 2)
                cv2.circle(frame, (f10, f11), 5, (0, 255, 0), -1)

            
            cv2.imshow('Dense Optical Flow',bgr_dense_frame)
            #cv2.imshow('Sparse Optical Flow',frame)
            #cv2.waitKey(100)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
