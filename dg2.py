import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Python Imaging Library (PIL) - external library adds support for image processing capabilities
from PIL import Image
import PIL.ImageOps
import os,ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


X, y = fetch_openml('mnist_784',version = 1,return_X_y = True)
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)

xtrain , xtest ,ytrain , ytest = train_test_split(X,y,random_state = 9 , train_size = 7500 , test_size = 2500)
xtrainScaled = xtrain / 255.0
xtestScaled = xtest / 255.0
clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(xtrainScaled , ytrain)
ypred = clf.predict(xtestScaled)
acc = accuracy_score(ypred , ytest )

cap = cv2.VideoCapture(0)
while(True):
    try:
     ret , frame = cap.read()
     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     height,width = gray.shape
     upperleft = (int(width/2 - 56),int(height/2 - 56))
     bottomright = (int(width/2 + 56),int(height/2 + 56))
     cv2.rectangle(gray,upperleft,bottomright,(0,255,0), 2 )
     
     #roi = Region Of Interest/focus area
     roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]] 
     impil = Image.fromarray(roi)
     imgbw = impil.convert('L')
     imgbwresize = imgbw.resize((28,28),Image.ANTIALIAS)
     imginverted = PIL.ImageOps.invert(imgbwresize) 
     pixelfilter = 20

     #percentile() converts the values in scalar quantity
     minpixel = np.percentile(imginverted , pixelfilter)

     #using clip to limit the values betwn 0-255
     imginvertedscale = np.clip(imginverted - minpixel ,0 ,255)
     maxpixel = np.max(imginverted)
     imginvertedscale = np.asarray(imginvertedscale)/maxpixel
     testsample = np.array(imginvertedscale).reshape(1,784)
     testpred = clf.predict(testsample)
     print("The predicted class is :" , testpred)
     cv2.imshow("frame",gray)
     if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()



