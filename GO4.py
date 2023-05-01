"""
Example code using the judge module
"""
import time

# pylint: disable=import-error
import cv2
import keyboard
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import datetime

from machathon_judge import Simulator, Judge

class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    if (slope == 0):
        return None

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def getROI(image): ##IT'S NOT SYMMETRIC AND CHARACTARIZED BASED ON MANY TRAILS AND ERRORS##

    # Defining Triangular ROI: The values will change as per your camera mounts
    polygon = np.array([[(100, 460),(110,430),(310,350),(560,430),(520,460)]])

    # creating black image same as that of input imag
    black_image = np.zeros_like(image)
    # Put the Triangular shape on top of our Black image to create a mask
    mask = cv2.fillPoly(black_image, polygon, 255)
    # applying mask on original image
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def run_car(simulator: Simulator) -> None:
    
    """
    Function to control the car using keyboard

    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """
    fps_counter.step()

    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_R = cv2.threshold(img[:,:,1], 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    fps = fps_counter.get_fps()
    # draw fps on image
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Specify kernel size and sigma for The gaussian filter
    kernel_size = (5, 5)
    sigma = 1
    blurred = cv2.GaussianBlur(img_R, kernel_size, sigma)
    canny= cv2.Canny(image=blurred, threshold1=40, threshold2=220, apertureSize=3, L2gradient=True)
    roi_img= getROI(canny)
    linesP = cv2.HoughLinesP(image=roi_img, rho=0.85, theta=np.pi*1.5/180, threshold=35, minLineLength=0, maxLineGap=600)

    res = np.copy(img)
    res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)
    Linr= lane_lines(res, linesP)

    slope1 = 0
    slope2 = 0
    avgslope = 0
    if Linr is not None:
            if Linr[0] is not None:
                N1,N2 = Linr[0]
                x1,y1 = N1
                x2,y2 = N2
                slope1 = (x2-x1)/(y2-y1)
                cv2.line(res, (x1, y1), (x2, y2), (0,255,0), 3, cv2.LINE_AA)
    if Linr is not None:
            if Linr[1] is not None:
                N1,N2 = Linr[1]
                x1,y1 = N1
                x2,y2 = N2
                cv2.line(res, (x1, y1), (x2, y2), (0,255,0), 3, cv2.LINE_AA)
                slope2 = (x2-x1)/(y2-y1)

    flag1 = False
    flag2 = False

    if slope2 !=0 and slope1!=0 :
        if(abs(slope1)<abs(slope2)):
            avgslope = slope1
            flag1 = True 
        if(abs(slope2)<abs(slope1) ):
            avgslope = slope2 
            flag2 = True

    else:
        if(slope1 == 0):
            avgslope = slope2
            flag2 = True
        if(slope2 == 0):
            avgslope = slope1
            flag1 = True

    e_current = 0
    if flag1:
            e_current = 2 - avgslope           # current error
    if flag2:
            e_current = avgslope  + 2
            e_current = -e_current

    cv2.line(res, (0, 480), (680, 480),(0,0,255), 3 )
    cv2.imshow("out",res)
    cv2.waitKey(1) 

    #####################################
    ######## HARD CODDED CONTROL ########
    #####################################
    #### THE PARAMETERS ARE RANDOM AND BASED ON MANY TRIAL AND ERROR ####

    throttle = 1
    loet = np.log(abs(e_current)) ##Steering coeff for mid slopes
    poet = pow(0.8,abs(e_current)) ##Steering coeff for smallest slopes

    if e_current>0 and slope1<0:    ####STEERING RIGHT####

        if 0 < e_current < 3.9: ## is slope is small
            steering = -0.86*poet
            throttle = 0.58*pow(0.2,abs(steering)) ##THROTTLE FUNCTION OF STEERING##
            simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
            simulator.set_car_velocity(throttle*35)

        elif 6.5 > e_current > 3.9: ## if slope is mid
            steering = -0.57*loet
            throttle = 0.72*pow(0.2,abs(steering))
            simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
            simulator.set_car_velocity(throttle*29)

        elif 6.5 < e_current: ## if slope is high
            steering = -1.18
            throttle = 1
            simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
            simulator.set_car_velocity(throttle*40)

    elif e_current<0 and slope2>0:  ####STEERING LEFT####

        if 0 > e_current > -3.9:
            steering = 0.86*poet
            throttle = 0.58*pow(0.2,abs(steering))
            simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
            simulator.set_car_velocity(throttle*35)

        elif -6.5 < e_current < -3.9:
            steering = 0.57*loet
            throttle = 0.72*pow(0.2,abs(steering))
            simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
            simulator.set_car_velocity(throttle*27)

        elif -6.5 > e_current:
            steering = 1.18
            throttle = 1
            simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
            simulator.set_car_velocity(throttle*40)

    else: ## if there is no detected lines
        throttle = 1
        steering = 0
        simulator.set_car_steering(steering * simulator.max_steer_angle / 1.65)
        simulator.set_car_velocity(throttle*40)


if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="CODE", zip_file_path="PATH")
    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
