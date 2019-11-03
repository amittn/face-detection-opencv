import numpy as np
import cv2
import matplotlib.pyplot as plt



#Loading the image to be tested
#test_image = cv2.imread('test8.jpg')

#Converting to grayscale
#test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Displaying the grayscale image
#plt.imshow(test_image_gray, cmap='gray')

#Since we know that OpenCV loads an image in BGR format, so we need to convert it into RBG format to
# be able to display its true colors. Let us write a small function for that.

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print('------------1------------')


def convertToRGB(image):
    print('-----------4-------------')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)

    print('-----------3-------------')
    return image_copy


#loading image
test_image2 = cv2.imread('test8.jpg')

#Converting to grayscale
test_image_gray = cv2.cvtColor(test_image2, cv2.COLOR_BGR2GRAY)

#Displaying grayscale image
plt.imshow(test_image_gray, cmap='gray')

#call the function to detect faces
print('-----------2-------------')
faces = detect_faces(haar_cascade_face, test_image2)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))
plt.show()
print('------------5------------')
key = cv2.waitKey(0)

if (key == 113 or key == 27):
    # exit is q or ESC is pressed
    #cv2.destroyAllWindows()
    exit()