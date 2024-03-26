#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt #Creating static,animated,and interactive visualizations
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np #used to create, manipulate, and analyze NumPy arrays


# In[4]:


#https://keras.io/api/datasets/mnist/
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[5]:


len(X_train)#60000 images


# In[6]:


len(X_test)#10000 images


# In[11]:


X_train[0].shape#28,28


# In[17]:


X_train[999]#28,28 array


# In[18]:


plt.matshow(X_train[999])


# In[25]:


y_train[999]


# In[23]:


y_train[:100]


# In[27]:


X_train = X_train / 255
X_test = X_test / 255
#pixels value-->0 to 255


# In[28]:


X_train[0]


# In[29]:


#2D to 1D(Flattening)
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[30]:


X_train_flattened.shape#28*28=784


# In[39]:


X_train_flattened[0]#In single


# In[40]:


#sequential- Stack of layers. Accepts every layer as an element
#Dense-all neurons connected 
#In sparse_categorical_crossentropy, categorical means output class is categorical
#In sparse_categorical_crossentropy, sparse means output variable is actually an integer
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=50)
#Training happens using model.fit


# In[45]:


model.evaluate(X_test_flattened, y_test)


# In[48]:


y_predicted = model.predict(X_test_flattened)
y_predicted[5]


# In[50]:


plt.matshow(X_test[5])


# In[51]:


np.argmax(y_predicted[5])


# In[53]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[54]:


y_predicted_labels[:10]


# In[55]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[56]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[57]:


#Using hidden layer
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=50)


# In[ ]:


'''#adding more hidden layers 

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(50, input_shape=(100,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=50)'''


# In[58]:


model.evaluate(X_test_flattened,y_test)


# In[59]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


USING FLATTEN LAYER


# In[60]:


#Using Flatten layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)


# In[61]:


model.evaluate(X_test,y_test)


# In[65]:


y_predicted = model.predict(X_test_flattened)


# In[66]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[67]:


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

