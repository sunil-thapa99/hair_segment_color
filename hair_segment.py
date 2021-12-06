#!/usr/bin/env python
# coding: utf-8

# In[95]:


from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from keras.models import load_model
import numpy as np
import cv2
from dnn_face_detection import detect_face


# In[96]:


model = load_model('model.h5')


# In[3]:


def predict(image):
  return model.predict(np.asarray([image]) ).reshape((224,224))

def display(img):
    plt.imshow(img,cmap='gray')
    plt.show()


# In[190]:


img = cv2.imread('1.jpg')
bounding_box = detect_face(img)
print(bounding_box)

for box in bounding_box:
    (x,y,w,h) = box.astype("int")
    x -= 200
    y -= 200
    w += 100

    crop_face = img[0:h, 0:img.shape[1]]
    
    print(x,y,w,h)


# In[191]:


display(crop_face)


# In[192]:


res_img = cv2.resize(crop_face, (224, 224))
img = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)


# In[193]:


display(res_img)


# In[194]:


display(img)


# In[195]:


pred_image = resize(img,(224,224)).reshape((224,224,1))
pred = predict(pred_image)
display(pred)


# In[196]:


# pred_image = resize(img,(224,224)).reshape((224,224,1))
pred_img = cv2.resize(img, (224, 224)).reshape((224, 224, 1))
pred2 = predict(pred_img)
display(pred2)


# In[197]:


rgb_img = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
# rgb_img[pred2>0.5] = (255, 0, 0)
# rgb_img[pred2<=0.5] = (0, 0, 0)
rgb_img[pred>0.5] = (105, 105, 105)


# In[198]:


norm_image = cv2.normalize(rgb_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
norm_image = norm_image.astype(np.uint8)


# In[199]:


display(norm_image)


# In[200]:


# combined = cv2.bitwise_and(res_img, res_img, mask=norm_image)
new_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
# combined = cv2.add(norm_image, new_img)
# combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
combined = cv2.addWeighted(new_img, 1, norm_image, 0.3, 0)


# In[201]:



display(combined)


# In[ ]:




