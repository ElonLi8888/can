"""This program uses OpenCV to detect circles in an image. It applies Gaussian blur to reduce noise and then uses Hough Circle Transform to detect circular shapes in the image. 
The program starts by masking the center of the image and incrementally increases the radius to detect circles within that mask. 
Once a circle is detected, the program highlights the circle and its center and displays the result. The process stops after detecting the first circle."""

import cv2
import numpy as np

# Load the image  
image_path = 'can.png'  
image = cv2.imread(image_path)  
if image is None:  
    raise FileNotFoundError(f"Error: Could not load image at '{image_path}'.")  # Ensure the image loads correctly  

image = image.copy()  # Create a copy to avoid modifying the original  

# Convert to grayscale  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grayscale conversion improves edge detection  

# Apply Gaussian blur  
blurred = cv2.GaussianBlur(gray, (9, 9), 2)  # Reduces noise and smooths the image for better circle detection  

# Get image dimensions  
height, width = gray.shape  
center_x, center_y = width // 2, height // 2  # Determine the center of the image  

# Initialize parameters for Hough Circle Transform  
radius = 0
param1 = 50
param2 = 30  
dp = 1.2
minDist = 20
min_radius = 10
max_radius = 0

# Start from the center and create outward masks  
while True:  
    radius += 1  # Increment radius to expand the mask outward  

    # Create circular mask at each radius  
    mask = np.zeros_like(gray)  
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)  # Draw a filled circle on the mask  

    # Apply the mask to the image  
    masked_image = cv2.bitwise_and(image, image, mask=mask)  

    # Detect circles using Hough Transform  
    circles = cv2.HoughCircles(  
        blurred, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2,  
        minRadius=min_radius, maxRadius=max_radius  
    )  

    if circles is not None:  # If circles are detected, process them  
        circles = np.uint16(np.around(circles))  # Convert detected circle values to integers  
        for circle in circles[0, :]:  
            x, y, detected_radius = circle  

            # Draw the detected circle and its center  
            cv2.circle(image, (x, y), detected_radius, (0, 255, 0), 3)  # Green circle outline  
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # Red dot at the center  

            # Display the image with the detected circle  
            cv2.imshow("Detected Circle", image)  
            cv2.waitKey(0)  
            cv2.destroyAllWindows()  # Close the image window after key press  
            break  # Stop after detecting the first circle  

    else:  
        print("No circle detected")  # Print a message if no circle is found  

    if radius >= gray.shape[0]:  # Stop if the mask grows larger than the image  
        break  
