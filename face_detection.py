import cv2
import argparse

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_faces(cascade, test_image, scaleFactor = 1.3):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 5)

    return image_copy


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the image file")
    args = vars(ap.parse_args())
    return args


if __name__ == "__main__":
    args = parse_arguments()

    # loading image
    test_image = cv2.imread(args['image'])

    # call the function to detect faces
    return_image = detect_faces(haar_cascade_face, test_image)

    cv2.imshow('Final', return_image)
    key = cv2.waitKey(0)

    if (key == 113 or key == 27):
        # exit is q or ESC is pressed
        cv2.destroyAllWindows()
        exit()