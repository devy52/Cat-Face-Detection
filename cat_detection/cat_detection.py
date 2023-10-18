import cv2
import matplotlib.pyplot as plt

# Ask the user to select the cat image file
image_path = "C:/Users/hp/Downloads/cat_detection/sample img/cat1.jpg"

if image_path:
    # Load input image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load cat detector haar cascade, then detect cat faces in the input image
    detector = cv2.CascadeClassifier("haarcascade_file/haarcascade_frontalcatface_extended.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1, minSize=(10, 10))
    print('No. of cats detected: ',len(faces))

    # if atleast one cat is detected
    if len(faces)>0:
        # Loop over cat faces and draw a rectangle on each face
        for ((x, y, w, h)) in (faces):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, f'Cat', (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Show the result using Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis
    plt.show()
else:
    print("No image path specified")


