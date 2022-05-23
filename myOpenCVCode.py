# Handwritten Digits recognition for OpenCV Python Project
# Coded by: Ryan Cummings 015933541 CMPE 258
# Status: Release 
# Date: Apr 10, 2022 


if __name__ == '__main__':

    import cv2
    from keras.models import load_model
    from keras.preprocessing import image
    import numpy as np
    import math
    from PIL import Image
    import os

    # This is the main function that takes the image/frame and does the ROI and Prediction
    def roi_and_preds(img):

        img = cv2.resize(img, (512, 512))
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray scale image', imgray)
        ret, binary_img = cv2.threshold(imgray,100,255,cv2.THRESH_BINARY)
        #cv2.imshow("Binary Image", binary_img)
        img_neg = cv2.bitwise_not(binary_img)
        #cv2.imshow("Negative Image", img_neg)
        thresh = cv2.Canny(imgray, 100, 200)
        #cv2.imshow('Canny', thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contour
        thresh1 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 5) # all contours
        #cv2.imshow('All contours', thresh1)
        
        #Bounding box
        for i in range(len(contours)):
            [x, y, w, h] = cv2.boundingRect(contours[i])
            #print("i:",i)
            if ((w*h) < 2000) | ((w*h)>28000):
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.imshow('i'+str(i), img)
            
            # crop binary_img to just the digit
            cropped_img = img_neg[y:y+h, x:x+w]#img[y:y+h, x:x+w]
            #cv2.imshow('cropped'+str(i), cropped_img)
            
            # place cropped on top of a square (max(w,h))
            cropped_img = cropped_img.copy()
            h, w = cropped_img.shape[:2]
            vis = cv2.copyMakeBorder(cropped_img, 50, 50, round(h/3)+round(h/5), round(h/3)+round(h/5), cv2.BORDER_CONSTANT)
            cv2.imshow('Preprocessed Square Image (before resizing):'+str(i),vis)

            # then reduce the size to 28x28
            vis_resized = cv2.resize(vis, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('vis resized'+str(i),vis_resized)
            im2arr = np.array(vis_resized)
            im2arr = im2arr.reshape(1,28*28)

            # doing the prediction on the image:
            y_pred = model.predict(im2arr)
            print(y_pred)
            print(np.argmax(y_pred))

            # display the digit on top of the bounding box in img
            cv2.putText(img, 'Predict: '+str(np.argmax(y_pred)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 1)
        
        # displaying img with all bounding boxes'
        cv2.imshow('sorted', img)
        return img


    # Loading the MNIST Model provided to us by Prof:
    model = load_model('mnist.h5')
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    # Starting the Program:
    input_selection = input('\n\n\n\nHow would you like to run the prediction? Live input from webcam (Webcam) or from a saved video file (Video)? Type: Webcam/Video\n')

    ####################### WEBCAM ROI AND PREDICTIONS ########################
    if input_selection == 'Webcam':
        #print("This is from webcam")
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            roi_and_preds(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    ####################### VIDEO FILE ROI AND PREDICTIONS ####################
    elif input_selection == 'Video':
        file_name = input('Enter video file name: ')

        # image = cv2.imread(img, cv2.IMREAD_COLOR)
        cap = cv2.VideoCapture(file_name)

        if (cap.isOpened() == False): 
            print("Error opening video/image file")
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                #Displaying each from and doing the manipulations:
                roi_and_preds(frame)
                cv2.waitKey(1)
                
            # Else: break out of loop
            else: 
                break
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()


    ####################### TESTING FOR STOCK IMAGE ROI AND PREDICTIONS ###################
    elif input_selection == 'Test':
        image = input('Enter image file name: ')
        roi_and_preds(cv2.imread(image, cv2.IMREAD_COLOR))

    else:
        print("Input command invalid, please re-run this program again with the commands: Video or Webcam.")
        exit()


cv2.waitKey(0)
cv2.destroyAllWindows()