import cv2
import numpy as np

def correlate_func(template,img):
    """
    Computes the correlation to find the best match of a template in the input image img

    Args:
        template (numpy array): template to scan over the image
        img (numpy array): the image to find the template at

    Returns:
        float, tuple: the floating value for the best fitted and the location of the best matched of the template
    """
    
    # Convert to gray scale
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Computing the normalized cross-correlation
    template_mean = template.mean()
    img_mean = img.mean()
    template_centered = template - template_mean
    img_centered = img - img_mean
     
    # Compute the sum of the squares of the template and the image
    template_sq = np.sum(template_centered**2)
    img_sq = np.sum(img_centered**2)
    # Compute the product of the template and the image
    product = np.zeros(img.shape)
    for y in range(template.shape[0]):
        for x in range(template.shape[1]):
            product[y:img.shape[0]-template.shape[0]+y+1, x:img.shape[1]-template.shape[1]+x+1] += template_centered[y,x] * img_centered[y:img.shape[0]-template.shape[0]+y+1, x:img.shape[1]-template.shape[1]+x+1]
    # Compute the normalized cross-correlation between the template and the image
    corr = product / np.sqrt(template_sq * img_sq)
    # Find the location of the maximum correlation in the correlation matrix
    max_loc = np.unravel_index(corr.argmax(), corr.shape)
    return corr[max_loc], max_loc

def img_sum_diff(input_img1, input_img2):
    """_summary_
    Takes two equal size images and computes the sum of absolute differences

    Args:
        input_img1 (numpy array): input image 1
        input_img2 (numpy array): input image 2

    Returns:
        numpy array: sum of absolute differences of the 2 images
    """
    # Converts the image into grayscale 
    #input_img1 = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
    #input_img2 = cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
    
    # Checking for images are equal size
    assert input_img1.size == input_img2.size, f"Size of image 1 {input_img1.size} must match image 2 {input_img2.size}"
    
    # Computes the sum of absolute differences
    abs_diff = np.abs(np.subtract(input_img1.astype(float), input_img2.astype(float)))
    sad = np.sum(abs_diff)
    
    return sad

def scan_for_lowest_disparity(img1, img2):
    """
    Loops through image 2 from left to right with and computes the 
    sum of absolute differences with image 1 

    Args:
        img1 (numpy array): image 2 
        img2 (numpy array): image 1

    Returns:
        int: The lowest matching disparity and the x location in image 2 where image 1 best fits
    """
    
    # Checking if the images has the same height 
    assert img1.shape[0] == img2.shape[0], f"Height {img1.shape[0]} of image 1 must match Height {img2.shape[0]} of image 2 "
    old_res = 1000000
    # Scanning from left to right
    for x in range(img2.shape[1] - img1.shape[1] + 1):
        new_res = img_sum_diff(img1,img2[:,x:x+img1.shape[1]])
        if new_res <= old_res:
            old_res = new_res
            res_x = x

    return old_res, res_x

def template_matching_inter(left_img, right_img):
    """
    Template matching function that iteratively takes a 7x7 subpart of left img and 
    uses the scan_for_lowest_disparity function on the corresponding row of right img.
    Then stores the best matching in a result list. Continue until all possible templates for a row has been
    used then move to next row.

    Args:
        left_img (numpy array): left image 
        right_img (numpy array): right image 
    """
    
    # Converts the image into grayscale 
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    res_list = []
    # Interatively takes 7x7 subparts
    for y in range(left_img.shape[0]-6):
        for x in range(left_img.shape[1]-6):
            subpart = left_img[y:y+7, x:x+7]
            corresponding_row = right_img[y:y+7,:]
            res,_ = scan_for_best_loc(subpart, corresponding_row)
            res_list.append(res)
            
    print(res_list)
    
def main():
    """
    Main function
    """
    nose_left = cv2.imread('nose_left.png')
    nose_right = cv2.imread('nose_right.png')
    nose_1 = cv2.imread('nose1.png')
    nose_2 = cv2.imread('nose2.png')
    nose_3 = cv2.imread('nose3.png')
    nose_span = cv2.imread('nose_span.png')
    
    tsukuba_left = cv2.imread('tsukuba_left.png')
    tsukuba_right = cv2.imread('tsukuba_right.png')
    
    # Best fitted for nose left
    nose_left_dict = {'nose left vs nose1': img_sum_diff(nose_left, nose_1), 
                      'nose left vs nose2': img_sum_diff(nose_left, nose_2), 
                      'nose left vs nose3': img_sum_diff(nose_left, nose_3)}
    
    smallest_val = min(nose_left_dict.values())
    smallest_vars = [key for key, val in nose_left_dict.items() if val == smallest_val]
    print(f"Best SAD is {smallest_vars} with a value of {smallest_val}")
    
    # The best fitted location and lowest matching disparity
    template_val_scan,template_loc_scan = scan_for_lowest_disparity(nose_left, nose_span)
    print(f"Location using scanning method: {template_loc_scan} with a value of {template_val_scan}")
    # Draws the best location on the nose span
    cv2.circle(nose_span, (template_loc_scan,3), 5, (0,0,255), -1)
    # cv2.imshow("Best loc", nose_span)



    # Template matching interatively
    template_matching_inter(tsukuba_left,tsukuba_right)
    


    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
        

    