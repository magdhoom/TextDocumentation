import cv2
import numpy as np
import math as m
import webbrowser

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("shot.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence>0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([center_x,center_y,x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
label=[]
identi=[]
for i in range(len(boxes)):
    if i in indexes:
        c_x,c_y,x, y, w, h = boxes[i]
        top=1
        label.append(class_ids[i])
        lb=label.count(class_ids[i])
        
        """
        while str(classes[class_ids[i]]) in label:
            classes[class_ids[i]]=classes[class_ids[i]]+str(top)
            print(str(classes[class_ids[i]]))
            top=top+1
            print(i,top)
        """
        #label.append(str(classes[class_ids[i]]))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(classes[class_ids[i]])+str(lb), (x, y + 30), font, 3, color, 3)
        identi.append(str(classes[class_ids[i]])+str(lb))

#Handwritten code,Distance calculation        
d_val=[]
indexes=indexes.tolist()
r_x=boxes[indexes[0][0]][0]
r_y=boxes[indexes[0][0]][1]
for i in range(1,len(indexes)):
    x=boxes[indexes[i][0]][0]
    y=boxes[indexes[i][0]][1]
    d=m.sqrt(((r_x-x)**2)+((r_y-y)**2))
    d_val.append(d)
#print("The distance values are ")
#print(d_val)
#print("The indexes are")
#print(indexes)

#Direction
di=[]
for i in range(1,len(indexes)):
    if (r_x>boxes[indexes[i][0]][0] and r_y>boxes[indexes[i][0]][1]):
        di.append("top_left")
    elif (r_x<boxes[indexes[i][0]][0] and r_y<boxes[indexes[i][0]][1]):
        di.append("bottom_right")
    elif (r_x<boxes[indexes[i][0]][0] and r_y>boxes[indexes[i][0]][1]):
        di.append("top_right")
    elif (r_x>boxes[indexes[i][0]][0] and r_y<boxes[indexes[i][0]][1]):
        di.append("bottom_left")
    elif (r_x==boxes[indexes[i][0]][0] and r_y<boxes[indexes[i][0]][1]):
        di.append("bottom")
    elif (r_x==boxes[indexes[i][0]][0] and r_y>boxes[indexes[i][0]][1]):
        di.append("top")
    elif (r_x>boxes[indexes[i][0]][0] and r_y==boxes[indexes[i][0]][1]):    
        di.append("left")
    else:
        di.append("right")

#Document Preparation
file=open('new.txt','w')
file.write("\t\t\tThe objects found in the image are:\n\n")
for i in range(0,len(indexes)):
    file.write("\t\t\t\t\t* "+str(identi[i])+"\n")
file.write("\n\t\t\tHere we are considering "+str(identi[0])+" as the reference object.\n\n\n")
for i in range(0,len(indexes)-1):
    file.write("\t-->The object "+str(identi[i+1])+" is situated from the "+str(identi[0])+" at "+str(d_val[i])+" pixels at "+di[i]+" direction")
    file.write("\n\n")
file.close()

cv2.imshow("Image", img)
webbrowser.open("new.txt")
cv2.waitKey(0)
cv2.destroyAllWindows()

