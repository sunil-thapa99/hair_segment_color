import numpy as np
import cv2

modelFile = "models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face(img,blob_size = 600, threshold=0.5):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img,1.0, (blob_size, blob_size), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    boxes = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > threshold:
            box = faces[0, 0, i, 3:7]* np.array([width, height, width, height])
    
            if (0 <= box[0] <= width) and (0 <= box[1] <= height) and \
                (0 <= box[2] <= width) and (0 <= box[3] <= height):
                bb = np.round(box,2)
                # bb = np.append(bb, confidence)
                boxes.append(bb)

    return np.array(boxes)


if __name__ == '__main__':
    # img = cv2.imread("test/Rajendra K.C_[0.9595654].jpg")
    cap = cv2.VideoCapture('/media/info/New Volume/ComputerVision/attendance/app/datasets/20191115_152635.mp4')
    import time 
    stime = time.time()
    try:
        while True:
            ret, img = cap.read()
            boxes = detect_face(img)
            print(boxes)
            for box in boxes:
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
            nrof_faces = boxes.shape[0]
            # print(nrof_faces)
            cv2.imshow(" ", img)
            cv2.waitKey(1)
    except:
        pass
    print(time.time() - stime)