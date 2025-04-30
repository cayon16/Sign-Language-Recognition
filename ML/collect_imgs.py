import os

import cv2


DATA_DIR = "D:/HUST/20242/CV/create_dataset/sign-language-detector-python/dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 16
dataset_size = 2000

alphabet = {
    0: "apple", 1: "can", 2: "get", 3: "good", 4: "have", 5: "help", 6: "how",
    7: "I", 8: "like", 9: "love",    10: "my", 11: "no", 12: "null",
    13: "sorry", 14: "thank-you", 15: "want"
}

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, alphabet[j])):
        os.makedirs(os.path.join(DATA_DIR, alphabet[j]))

    print('Collecting data for class {}'.format(alphabet[j]))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, alphabet[j], '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()