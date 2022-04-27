# self-driving-car-CompVision

## Table of Contents

- [Project Description](#Project-Description)
- [ImplementationPhase1](#ImplementationPhase1)

## Project-Description
In this project we are going to create a simple perception stack for self-driving cars (SDCs.) Although a typical perception stack for a self-driving car may contain different data sources from different sensors (ex.: cameras, lidar, radar, etc…), we’re only going to be focusing on video streams from cameras for simplicity. We’re mainly going to be analyzing the road ahead, detecting the lane lines, detecting other cars/agents on the road, and estimating some useful information that may help other SDCs stacks. The project is split into two phases. 
### The First phase (Detect line in images and videos)
The expected output is as follows:
1) Your pipeline should be able to detect the lanes, highlight them using a fixed color, and pain the area between them in any color you like (it’s painted green in the image above.)
2) You’re required to be able to roughly estimate the vehicle position away from the center of the lane.
 #### ImplementationPhase1 
1) we use hough transform to detect the lines that we will draw of on the lanes
2) We could find the lines when we know the equation of line (the slope and interecpt)
3) We could find the region of interest and fill it by knowing the the equations of the lines to know the coordinates of the lines
4) We mask the yellow color and threshold the while color with the original image or video to detect the yellow and white lanes
![The result img!](resultImg.PNG)
