# HandWritten Digit Recognition

#### Ryan Cummings, 015933541, CMPE 258, Apr 10 2022

<hr>

# Welcome to my CMPE 258 Project "Handwritten Digit Recognition":

## In this project we were asked to create a handwritten digit recognition tool with OpenCV in Python.

## My system is able to: 1) Display a bounding box on each individual digit 2) Display preprocessed square image before resizing to 28x28 and 3) Feed the digit to mnist CNN for recognition and displays the predicted digit on top of it's bounding box!

# There are a total of 2 operations you can do with my program:
## 1) Run the handwritten digit recognition with your webcam.
Navigate to the directory of this program, then type: 

    python myOpenCVCode.py

Then, when prompted, type:

    Webcam

#### From there you will need to hold up your handwritten digits to the webcam! (Be sure to move the displayed video window so you can see the processed square images)
 
## 2) Run the handwritten digit recognition with a saved video file.
    python myOpenCVCode.py
Then when prompted type:
    
    Video

Then, when prompted, type the name of the saved video file:

    digits_video.mp4

#### You will then see the handwritten digit recognition running on the saved video file! (Be sure to move the displayed video window so you can see the processed square images)

<hr>
# You can view my processed video and webcam results by watching the processed_digits.mp4 and processed_webcam.mp4 respectively.

<hr>
# Addressing bugs with the program:
1) There seems to be an issue when your write the digit too skinny or too wide the mnist.h5 model does not predict it well.
2) On my end, the webcam and video processing run kind of slow but still at a moderate pace... this is an issue with my hardware since I am on a PC that is about 5-6 years old.
3) If there is a weird shadow over the piece of paper when you hold it up to the webcam there may be some boxes that are generated unintentionally.
