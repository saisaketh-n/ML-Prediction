#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


iris_data=pd.read_csv("IRIS.csv")


# In[3]:


iris_data.head()


# In[4]:


iris_data.tail()


# In[5]:


iris_data.shape


# In[6]:


iris_data.columns


# In[7]:


iris_data.rename(columns={'Sepal_length':'sepal_length','Sepal_width':'sepal_width','Petal_length':'petal_length','Petal_width':'petal_width','Species':'species'})


# In[8]:


iris_data.info()


# In[9]:


iris_data.isnull().sum()


# In[10]:


iris_data.describe()


# In[11]:


iris_data['species'].value_counts()


# In[12]:


sns.set_style("dark")
sns.countplot('species',data=iris_data)
plt.show()


# In[13]:


sns.boxplot(x='species',y='sepal_length',data=iris_data)
plt.show()


# In[14]:


sns.boxplot(x='species',y='sepal_width',data=iris_data)
plt.show()


# In[15]:


sns.boxplot(x='species',y='petal_length',data=iris_data)
plt.show()


# In[16]:


sns.boxplot(x='species',y='petal_width',data=iris_data)
plt.show()


# In[17]:


fig,axes=plt.subplots(2,2,figsize=(16,9))
sns.boxplot(ax=axes[0,0],data=iris_data,x='species',y='sepal_length')
sns.boxplot(ax=axes[0,1],data=iris_data,x='species',y='sepal_length')
sns.boxplot(ax=axes[1,0],data=iris_data,x='species',y='sepal_length')
sns.boxplot(ax=axes[1,1],data=iris_data,x='species',y='sepal_length')


# In[18]:


plt.figure(figsize=(15,9))
sns.scatterplot(x='sepal_length',y='sepal_width',hue='species',data=iris_data)
plt.show()


# In[19]:


sns.histplot(x ='sepal_length',hue='species',kde=True, data= iris_data,element="step")
plt.show()


# In[20]:


sns.histplot(x ='sepal_width',hue='species',kde=True, data= iris_data,element="step")
plt.show()


# In[21]:


sns.histplot(x ='petal_length',hue='species',kde=True, data= iris_data,element="step")
plt.show()


# In[22]:


sns.histplot(x ='petal_width',hue='species',kde=True, data= iris_data,element="step")
plt.show()


# In[23]:


fig, axes = plt.subplots(2, 2, figsize=(16,9))
sns.histplot(ax = axes[0,0],data=iris_data,x ='sepal_width',hue='species',kde=True)
sns.histplot(ax = axes[0,1],data=iris_data,x ='sepal_length',hue='species',kde=True)
sns.histplot(ax = axes[1,0],data=iris_data,x ='petal_width',hue='species',kde=True)
sns.histplot(ax = axes[1,1],data=iris_data,x ='petal_length',hue='species',kde=True)
plt.show()


# In[24]:


sns.pairplot(iris_data,hue="species",size=3);
plt.show()


# In[25]:


iris_data.corr().style.background_gradient(cmap='coolwarm').set_precision(2)


# In[26]:


sns.heatmap(iris_data.corr(),annot=True,cmap='Greens')
plt.show()


# In[27]:


X = iris_data.drop('species', axis = 1)
y = iris_data['species']


# In[28]:


X


# In[29]:


y


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=101)
print('The shape of X_train is: {}'.format(X_train.shape))
print('The shape of X_test is: {}'.format(X_test.shape))
print('The shape of y_train is: {}'.format(y_train.shape))
print('The shape of y_test is: {}'.format(y_test.shape))


# In[32]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[33]:


print(X_train)


# In[34]:


print(X_test)


# In[35]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[36]:


pred_train = model.predict(X_train)


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[38]:


pred_test = model.predict(X_test)


# In[39]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[40]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train,y_train)


# In[41]:


pred_train = model.predict(X_train)


# In[42]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[43]:


pred_test = model.predict(X_test)


# In[44]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[45]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)


# In[46]:


pred_train = model.predict(X_train)


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[48]:


pred_test = model.predict(X_test)


# In[49]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[50]:


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(X_train,y_train)


# In[51]:


pred_train = model.predict(X_train)


# In[52]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[53]:


pred_test = model.predict(X_test)


# In[54]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[55]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)


# In[56]:


pred_train = model.predict(X_train)


# In[57]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[58]:


pred_test = model.predict(X_test)


# In[59]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[60]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[61]:


pred_train = model.predict(X_train)


# In[62]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[63]:


pred_test = model.predict(X_test)


# In[64]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[65]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[66]:


pred_train = model.predict(X_train)


# In[67]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[68]:


pred_test = model.predict(X_test)


# In[69]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[70]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train,y_train)


# In[71]:


pred_train = model.predict(X_train)


# In[72]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[73]:


pred_test = model.predict(X_test)


# In[74]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[75]:


from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier()
model.fit(X_train,y_train)


# In[76]:


pred_train = model.predict(X_train)


# In[77]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[78]:


pred_test = model.predict(X_test)


# In[79]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[80]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train,y_train)


# In[81]:


pred_train = model.predict(X_train)


# In[82]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[83]:


pred_test = model.predict(X_test)


# In[84]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[85]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train,y_train)


# In[86]:


pred_train = model.predict(X_train)


# In[87]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_train,pred_train))
print(confusion_matrix(y_train,pred_train))
print('Accuracy Score of Model on train data is: {}' .format(accuracy_score(y_train,pred_train)))


# In[88]:


pred_test = model.predict(X_test)


# In[89]:


print(classification_report(y_test,pred_test))
print(confusion_matrix(y_test,pred_test))
print('Accuracy Score of Model on test data is: {}' .format(accuracy_score(y_test,pred_test)))


# In[ ]:




