import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

def canny(img):
    canny = cv.Canny(img, 50, 150)
    return canny
    
def region_of_interest(img):
    height = img.shape[0]
    poly = np.array([[(200,height), (1200, height), (790,450), (580, 450)]])
    mask = np.zeros_like(img)
    cv.fillPoly(mask, poly, 255)
    return mask
    
def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image
    
def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)
    
def make_line_points(y1, y2, line,old_slope):
    if line is None:
        return "not"
    
    slope, intercept = line
    #print("slope = ", slope)
    #print("intercept = ", intercept)
    
    # make sure everything is integer as cv2.line requires it
    x1=int((y1 - intercept)/old_slope) if slope ==0 else int((y1 - intercept)/slope)
    x2= int((y2 - intercept)/old_slope) if slope ==0 else int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))
    
def lane_lines(image, lines,left_slope,right_slope):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = 680 # bottom of the image
    y2 = y1*0.725     # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane,left_slope)
    right_line = make_line_points(y1-5, y2, right_lane,right_slope)
    return left_line, right_line,left_lane,right_lane

    
def draw_lane_lines(image, lines, color=[255,0, 0], thickness=10):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line != "not":
            cv.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv.addWeighted(image, 1.0, line_image, 0.8, 0.0)
    
def lane(left_line,right_line,img):
    poly = np.array([[(left_line[0][0],left_line[0][1]), (right_line[0][0], right_line[0][1]),  (right_line[1][0], right_line[1][1]),(left_line[1][0],left_line[1][1])]])
    mask = np.zeros_like(img)
    cv.fillPoly(mask, poly, color=(0, 255, 0))
    return mask

print ("Would you like to enter Debugging Mode?")
print("Press: [n] for no")
print("Press: Any other key for yes")
debug = str(input())

cap = cv.VideoCapture('project_video.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

left_line_old =((200,680),(790,450))
right_line_old=((1200,680),(580,450))
left_slope_old =100
right_slope_old=100

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Display the resulting frame
  
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)    
    low_yellow = np.uint8([ 10,   0, 100])
    up_yellow = np.uint8([ 40, 255, 255])
    mask = cv.inRange(hls, low_yellow, up_yellow)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), 0)
    interestt = blur & region_of_interest(blur)
    
    ret, thresh = cv.threshold(interestt, 210, 240, cv.THRESH_BINARY)
    
    new = mask | thresh
    
    interestt = new & region_of_interest(new)

    new_canny = cv.Canny(interestt, 50, 150)
    lines = cv.HoughLinesP(new_canny, 5, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=200)
    line_img = display_lines(frame, lines)
    
    left_line, right_line , left_slope, right_slope=lane_lines(frame, lines,left_slope_old,right_slope_old) 
    new_lines=left_line, right_line
    
    new_lines=list(new_lines)
    if  new_lines[0] == "not":
         new_lines[0]=left_line_old
    else:
        left_line_old= new_lines[0]

        
    if new_lines[1] == "not":
        new_lines[1]=right_line_old
    else:
        right_line_old=new_lines[1]
    
    if left_slope is not None:
        if left_slope[0] != 0:
            left_slope_old=left_slope[0]
    if right_slope is not None:
        if right_slope[0] !=0:
            right_slope_old=right_slope[0]
        
    
    lane_mask = lane( new_lines[0],new_lines[1],frame)

    x = draw_lane_lines(frame,new_lines)    
    lanes_img = cv.addWeighted(x, 1.0, lane_mask, 1.0, 0.0)
    
    
    lane_centre= ((new_lines[1][0][0]-new_lines[0][0][0])/2)+new_lines[0][0][0]
    width=frame.shape[1]
    car_centre=(width)/2
    diffrence=lane_centre-car_centre
    meters=diffrence*3/(new_lines[1][0][0]-new_lines[0][0][0])
    
    font = cv.FONT_HERSHEY_SIMPLEX
    meter =round(meters, 3)
    if meters >0:
        cv.putText(lanes_img,'Vehicle is '+ str(meter)+'m left of centre',(50, 50),font, 1, (255, 255, 255), 2, cv.LINE_4)
    elif meters<0:
        cv.putText(lanes_img,'Vehicle is '+ str(meter*-1)+'m right of centre',(50, 50),font, 1, (255, 255, 255), 2, cv.LINE_4)
    else: 
        cv.putText(lanes_img,'Vehicle is at centre',(50, 50),font, 1, (255, 255, 255), 2, cv.LINE_4)

    dim = (320,200)
    resized_interest= cv.resize(new_canny, dim)
    resized_thresh= cv.resize(interestt, dim)
    resized_mask= cv.resize(line_img, dim)
    resized_primary = cv.resize(lane_mask, dim)

    # Convert grayscale image to 3-channel image,so that they can be stacked together    
    resized_interest = cv.cvtColor(resized_interest,cv.COLOR_GRAY2BGR)
    resized_thresh= cv.cvtColor(resized_thresh,cv.COLOR_GRAY2BGR)
    #resized_primary = cv.cvtColor(resized_primary, cv.COLOR_GRAY2BGR)

    resized_interest = resized_interest[60:320, :]
    resized_thresh = resized_thresh[60:320, :]
    resized_primary = resized_primary[60:320, :]
    resized_mask = resized_mask[60:320, :]
    
    cv.putText(resized_interest,'Canny Edge Detection',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)
    cv.putText(resized_thresh,'Region of Interest',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)
    cv.putText(resized_primary,'Lane Fill',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)
    cv.putText(resized_mask,'Hough Lines',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)

    
    hori = np.concatenate((resized_interest, resized_thresh), axis = 1)
    hori = np.concatenate((hori, resized_mask), axis = 1)
    hori = np.concatenate((hori, resized_primary), axis = 1)
    vert = np.concatenate((lanes_img, hori), axis= 0)
    #resized_primary = cv.resize(vert, (1920,720))
    if (debug == 'n'):
        cv.imshow('Frame',lanes_img)
    else:
        cv.imshow('Frame',vert)
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()

# Reading an image
images_paths = glob.glob("images/*.jpg")
images = [cv.imread(image) for image in images_paths]

# getting img dimensions and printing it
image_idx = 0

#dummy
left_line_old =((200,680),(790,450))
right_line_old=((1200,680),(580,450))
left_slope_old =100
right_slope_old=100

for test_img in images:
    hls = cv.cvtColor(test_img, cv.COLOR_BGR2HLS)    
    low_yellow = np.uint8([ 10,   0, 100])
    up_yellow = np.uint8([ 40, 255, 255])
    mask = cv.inRange(hls, low_yellow, up_yellow)
    gray_img = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (3,3), 0)    
    blur_interest_img = blur_img & region_of_interest(blur_img)
    ret, thresh = cv.threshold(blur_interest_img, 210, 255, cv.THRESH_BINARY)
    two_lanes = mask | thresh
    interest_img = two_lanes & region_of_interest(two_lanes)
    new_canny = cv.Canny(interest_img, 50, 150)
    lines = cv.HoughLinesP(new_canny, 5, np.pi/180, 100, np.array([]), maxLineGap=300)
    line_img = display_lines(test_img, lines)
    left_line, right_line , left_slope, right_slope=lane_lines(test_img, lines,left_slope_old,right_slope_old) 
    new_lines=left_line, right_line
    new_lines=list(new_lines)
    if  new_lines[0] == "not":
         new_lines[0]=left_line_old
    else:
        left_line_old= new_lines[0]
      
    if new_lines[1] == "not":
        new_lines[1]=right_line_old
    else:
        right_line_old=new_lines[1]
    
    lane_centre= ((new_lines[1][0][0]-new_lines[0][0][0])/2)+new_lines[0][0][0]
    width=test_img.shape[1]
    car_centre=(width)/2
    diffrence=lane_centre-car_centre
    meters=diffrence*3/(new_lines[1][0][0]-new_lines[0][0][0])
    mask_lanes = lane(new_lines[0],new_lines[1],test_img)
    x = draw_lane_lines(test_img,new_lines)    
    lanes_img = cv.addWeighted(x, 1.0, mask_lanes, 0.8, 0.0)
    font = cv.FONT_HERSHEY_SIMPLEX
    meter =round(meters, 2)
    if meters >0:
        cv.putText(lanes_img,'Vehicle is '+ str(meter)+'m left of centre',(50, 50),font, 1, (255, 255, 255), 2, cv.LINE_4)
    elif meters<0:
        cv.putText(lanes_img,'Vehicle is '+ str(meter*-1)+'m right of centre',(50, 50),font, 1, (255, 255, 255), 2, cv.LINE_4)
    else:
        cv.putText(lanes_img,'Vehicle is at centre',(50, 50),font, 1, (255, 255, 255), 2, cv.LINE_4)
   
    
    dim = (320,200)
    resized_interest= cv.resize(new_canny, dim)
    resized_thresh= cv.resize(interest_img, dim)
    resized_mask= cv.resize(line_img, dim)
    resized_primary = cv.resize(mask_lanes, dim)

    # Convert grayscale image to 3-channel image,so that they can be stacked together    
    resized_interest = cv.cvtColor(resized_interest,cv.COLOR_GRAY2BGR)
    resized_thresh= cv.cvtColor(resized_thresh,cv.COLOR_GRAY2BGR)
    #resized_primary = cv.cvtColor(resized_primary, cv.COLOR_GRAY2BGR)

    resized_interest = resized_interest[60:320, :]
    resized_thresh = resized_thresh[60:320, :]
    resized_primary = resized_primary[60:320, :]
    resized_mask = resized_mask[60:320, :]
    
    cv.putText(resized_interest,'Canny Edge Detection',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)
    cv.putText(resized_thresh,'Region of Interest',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)
    cv.putText(resized_primary,'Lane Fill',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)
    cv.putText(resized_mask,'Hough Lines',(20, 30),font, 0.7, (255, 255, 255), 1, cv.LINE_4)

    
    hori = np.concatenate((resized_interest, resized_thresh), axis = 1)
    hori = np.concatenate((hori, resized_mask), axis = 1)
    hori = np.concatenate((hori, resized_primary), axis = 1)
    vert = np.concatenate((lanes_img, hori), axis= 0)
    
    cv.imwrite('output/test{}.jpg'.format(str(image_idx)), vert)
    image_idx = image_idx + 1