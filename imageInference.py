import torch
import cv2
import numpy as np



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('model3.pt', map_location=torch.device('cpu')).eval()

def bbox(detections,orig):
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > .9:
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format('mobile', confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(orig, (startX, startY), (endX, endY),(0,255,0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0), 2)
    return orig

path= './image/hp_104.jpg'# add path of the image

image = cv2.imread(path)
image = cv2.resize(image,(round(image.shape[1]*0.5), round(image.shape[0]*0.5)))
orig = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)
image = image.to(device)
detections = model(image)[0]
frame = bbox(detections,orig)
cv2.imshow('frame', frame)
