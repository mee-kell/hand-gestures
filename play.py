from keras.models import model_from_json
import numpy as np
from skimage import io
import cv2
import random

# Format image from camera


def prepImg(pth):
    return cv2.resize(pth, (300, 300)).reshape(1, 300, 300, 3)


# Load model for classification
with open('model.json', 'r') as f:
    loaded_model_json = f.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# One-Hot encoded representations of hand positions
shape_to_label = {
    'thumbsup': np.array([1., 0., 0.]),
    'circle': np.array([0., 1., 0.]),
    'victory': np.array([0., 0., 1.])
}
arr_to_shape = {np.argmax(shape_to_label[x]): x for x in shape_to_label.keys()}

# Start webcam and read each frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
loaded_model.predict(prepImg(frame[50:350, 100:400]))

NUM_ROUNDS = 5

# Start screen
while True:
    ret, frame = cap.read()
    frame = frame = cv2.putText(
        frame, "Press Space to start", (160, 200),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA
    )
    cv2.imshow('Hand Gestures', frame)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

# Play screen
for rounds in range(NUM_ROUNDS):
    pred = ""
    for i in range(90):
        ret, frame = cap.read()

        pred = arr_to_shape[np.argmax(
            loaded_model.predict(prepImg(frame[50:350, 100:400])))
        ]

        # Set images
        ''' if pred == "thumbsup":
            img = cv2.imread("./tigger.jpg")
            break '''

        # Explicitly draw target zone for image
        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)

        frame = cv2.putText(
            frame, pred, (150, 140), cv2.FONT_HERSHEY_SIMPLEX,
            1, (250, 250, 0), 2, cv2.LINE_AA
        )

        cv2.imshow('Hand Gestures', frame)

        # frame = cv2.imshow("Animal", img)
        # cv2.waitKey(1500)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

while True:

    ret, frame = cap.read()

    frame = cv2.putText(
        frame, "Press q to quit", (190, 200),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA
    )

    cv2.imshow('Hand Gestures', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
