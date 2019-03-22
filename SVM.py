#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import seaborn as sns
from sklearn import datasets


# In[3]:


df = sns.load_dataset('iris')


# In[4]:


df.head()


# In[5]:


col = ['petal_length','petal_width','species']


# In[6]:


df.loc[:,col].head()


# In[7]:


species_to_num = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['temp'] = df['species'].map(species_to_num)


# In[8]:


X = df.iloc[:,2:-2]
X.head()


# In[9]:


y = df['temp']
y.shape


# In[10]:


from sklearn.svm import SVC 


# In[11]:


svm_model = SVC(random_state=0, C = 0.5, gamma = 'auto', kernel='linear')


# In[12]:


svm_model.fit(X,y)


# In[13]:


svm_model.predict([[6,1]])


# In[14]:


Xv = X.values.reshape(-1,1)
h = .02
X_min, y_min = np.min(Xv), np.min(y)
X_max, y_max = np.max(Xv) + 1, np.max(y) + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max, h), np.arange(y_min, y_max, h))


# In[15]:


z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(16,10))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3);
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80, 
            alpha=0.9, edgecolors='g');


# In[16]:


X.head()


# In[17]:


y.tail()


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[19]:


from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)


# In[20]:


from sklearn.svm import SVC
svc = SVC(C = 1,kernel='linear')


# In[21]:


svc.fit(X_train,y_train)


# In[22]:


svc.score(X_train,y_train)


# # Cross validation Score within training sets

# In[23]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(svc, X_train, y_train, cv = 3)
score


# In[24]:


score.mean(), score.std()


# In[25]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(svc, X_train, y_train, cv =3)


# # Confusion Matrix

# In[26]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred)


# # Precison and recall score

# In[27]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_train, y_pred, average='macro'), recall_score(y_train, y_pred, average='macro'), f1_score(y_train, y_pred, average='macro')


# # Cross valdiation for test set

# In[28]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(svc, X_test, y_test, cv = 3)
score


# In[29]:


score.mean(), score.std()


# In[30]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(svc, X_test, y_test, cv =3)


# # Confusion Matrix

# In[31]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# # Precision recall score

# In[32]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')


# ***

# # Polynomial kernel in SVM

# In[235]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)


# In[236]:


from sklearn.svm import SVC
#svc = SVC(kernel='poly', C = 1, degree = 3, gamma='auto')
svc = SVC(kernel='rbf', C = 1, degree = 3, gamma=0.7)


# In[237]:


svc.fit(X, y)


# In[238]:


xv = X.values.reshape(-1,1)
h = .02
x_min,  x_max = np.min(xv), np.max(xv) + 1
y_min,  y_max = np.min(y), np.max(y) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# In[239]:


z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.figure(figsize=(12,8))
ax = plt.contourf(xx, yy, z, cmap = 'afmhot', alpha=0.3)
plt.scatter(X.values[:,0], X.values[:,1], c = y)
plt.show();


# In[240]:


svc.score(X_train,y_train)


# In[241]:


svc.score(X_test, y_test)


# # Cross validation  and precison recall score on training set

# In[242]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(svc, X_train, y_train, cv = 3)
score


# In[243]:


score.mean(), score.std()


# In[244]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(svc, X_train, y_train, cv =3)


# In[245]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_pred)


# In[246]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_train, y_pred, average='macro'), recall_score(y_train, y_pred, average='macro'), f1_score(y_train, y_pred, average='macro')


# # Cross validation  and precison recall score on test set

# In[247]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(svc, X_test, y_test, cv = 3)
score


# In[248]:


score.mean(), score.std()


# In[249]:


from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(svc, X_test, y_test, cv =3)


# In[250]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[251]:


from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred, average='macro'), recall_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='macro')


# # Grid Search

# In[252]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# In[253]:


pipeline = Pipeline([('clf', SVC(kernel='rbf', C=1, gamma=0.1))]) 


# In[254]:


params = {'clf__C':(0.1, 0.5, 1, 2, 5, 10, 20), 
          'clf__gamma':(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1)} 


# In[255]:


svm_grid_rbf = GridSearchCV(pipeline, params, n_jobs=-1, cv=3, verbose=1, scoring='accuracy',iid=False)


# In[256]:


svm_grid_rbf.fit(X_train, y_train)


# In[257]:


svm_grid_rbf.best_score_


# In[258]:


best = svm_grid_rbf.best_estimator_.get_params()
best


# In[259]:


for k in sorted(params.keys()):
    print("{0:} = {1:.4f}".format(k,best[k]))


# In[260]:


y_test_pred = svm_grid_rbf.predict(X_test)


# In[261]:


confusion_matrix(y_test, y_test_pred)


# In[262]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_test, 
                                                           y_test_pred, 
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                     y_test_pred, 
                                                     average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                             y_test_pred, 
                                             average='weighted')))


# In[ ]:





# In[ ]:




