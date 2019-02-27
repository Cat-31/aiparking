# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2


def rotate(p_image, angle, center=None, scale=1.0):
    (h, w) = p_image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(p_image, matrix, (w, h))

    return rotated


# load the image
print("[INFO] loading network...")
model = load_model('classify.h5')

label_dic = {0: "all free", 1: "left occupied", 2: 'right occupied', 3: 'fully occupied'}

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./videos/Live.mp4')

index = 0
idx = 0
while True:
    index += 1
    ret, image = cap.read()
    if index % 5 == 0:
        image = image[20:290, :, :]
        # image = rotate(image, -90)
        orig = image.copy()

        # pre-process the image for classification
        # cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite("./output/G{}.jpg".format(idx), image)
        image = cv2.resize(image, (100, 100))
        cv2.imshow("net input", image)
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        p = model.predict(image)[0]

        # build the label
        proba = p[int(np.argmax(p))]
        label = "{}: {:.2f}%".format(label_dic[int(np.argmax(p))], proba * 100)

        # draw the label on the image
        cv2.putText(orig, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # show the output image
        cv2.imshow("Parking", orig)

        # cv2.imwrite("./output/C{}.jpg".format(idx), orig)
        idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
