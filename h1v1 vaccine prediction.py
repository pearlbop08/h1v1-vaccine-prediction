#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


data=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv")


# # We need to tell whether the respondent received H1N1 flu vaccine or not
# ### hence it is a Classification problem because it says vaccine is received or not
# 

# ## Data Visualizing

# In[3]:


data.shape


# In[4]:


data.tail(20)


# In[5]:


duplicate=data.duplicated()
print(duplicate.sum())


# In[6]:


data.drop_duplicates(inplace=True)
data.duplicated().sum()


# In[7]:


plt.figure(figsize=(10,10))
sns.heatmap(data.isnull())


# In[8]:


data.boxplot()


# In[9]:


data.hist(figsize=(20,15))


# ## Data cleaning

# ### Here 0:not vaccinated ; 1: vaccinated

# In[10]:


data["h1n1_vaccine"].value_counts()


# In[11]:


data.groupby("h1n1_vaccine").mean()


# In[12]:


data.isnull().sum()


# In[13]:


null_var=data.isnull().sum()/data.shape[0]*1000
null_var


# In[14]:


data.dtypes


# In[15]:


drop_columns=null_var[null_var>50].keys()
drop_columns


# In[16]:


data.drop(['dr_recc_h1n1_vacc', 'dr_recc_seasonal_vacc', 'has_health_insur',
       'qualification', 'income_level', 'marital_status', 'housing_status',
       'employment'],axis=1,inplace=True)


# In[17]:


data.drop('unique_id',axis=1,inplace=True)


# In[18]:


data.shape


# In[19]:


data["age_bracket"]=data["age_bracket"].replace({1:"65+ Years", 2:"18 - 34 Years", 3:"55 - 64 Years",4:"35 - 44 Years", 5:"45 - 54 Years"})
data.sample(10)


# In[20]:


data=pd.get_dummies(data,columns=["age_bracket"])
data.sample(10)


# In[21]:


data["race"]=data["race"].replace({1:"White", 2:"Black", 3:"Hispanic",4:"Other or Multiple"})
data.sample(10)


# In[22]:


data=pd.get_dummies(data,columns=["race"])
data.sample(10)


# In[23]:


data["sex"]=data["sex"].replace({1:"Male", 2:"Female"})
data.sample(10)


# In[24]:


data=pd.get_dummies(data,columns=["sex"])
data.sample(10)


# In[25]:


data["census_msa"]=data["census_msa"].replace({1:"Non-MSA", 2:"MSA, Principle City", 3:"MSA, Not Principle City"})
data.sample(10)


# In[26]:


data=pd.get_dummies(data,columns=["census_msa"])
data.sample(10)


# In[27]:


data["avoid_touch_face"] = data["avoid_touch_face"].fillna(data["avoid_touch_face"].mean())


# In[28]:


data["reduced_outside_home_cont"] = data["reduced_outside_home_cont"].fillna(data["reduced_outside_home_cont"].mean())


# In[29]:


data["avoid_large_gatherings"] = data["avoid_large_gatherings"].fillna(data["avoid_large_gatherings"].mean())


# In[30]:


data["wash_hands_frequently"] = data["wash_hands_frequently"].fillna(data["wash_hands_frequently"].median())


# In[31]:


data["h1n1_worry"] = data["h1n1_worry"].fillna(data["h1n1_worry"].median())


# In[32]:


data["h1n1_awareness"] = data["h1n1_awareness"].fillna(data["h1n1_awareness"].median())


# In[33]:


data["antiviral_medication"] = data["antiviral_medication"].fillna(data["antiviral_medication"].median())


# In[34]:


data["bought_face_mask"] = data["bought_face_mask"].fillna(data["bought_face_mask"].mean())


# In[35]:


data["is_h1n1_vacc_effective"] = data["is_h1n1_vacc_effective"].fillna(data["is_h1n1_vacc_effective"].median())


# In[36]:


data["contact_avoidance"] = data["contact_avoidance"].fillna(data["contact_avoidance"].median())


# In[37]:


data["is_h1n1_vacc_effective"] = data["is_h1n1_vacc_effective"].fillna(data["is_h1n1_vacc_effective"].median())


# In[38]:


data["is_h1n1_risky"] = data["is_h1n1_risky"].fillna(data["is_h1n1_risky"].median())


# In[39]:


data["sick_from_h1n1_vacc"] = data["sick_from_h1n1_vacc"].fillna(data["sick_from_h1n1_vacc"].median())


# In[40]:


data["is_seas_vacc_effective"] = data["is_seas_vacc_effective"].fillna(data["is_seas_vacc_effective"].median())


# In[41]:


data["sick_from_h1n1_vacc"] = data["sick_from_h1n1_vacc"].fillna(data["sick_from_h1n1_vacc"].median())


# In[42]:


data["no_of_adults"] = data["no_of_adults"].fillna(data["no_of_adults"].median())


# In[43]:


data["no_of_children"] = data["no_of_children"].fillna(data["no_of_children"].median())


# In[44]:


data["is_seas_risky"] = data["is_seas_risky"].fillna(data["is_seas_risky"].median())


# In[45]:


data["sick_from_seas_vacc"] = data["sick_from_seas_vacc"].fillna(data["sick_from_seas_vacc"].median())


# In[46]:


data["chronic_medic_condition"] = data["chronic_medic_condition"].fillna(data["chronic_medic_condition"].median())


# In[47]:


data["cont_child_undr_6_mnths"] = data["cont_child_undr_6_mnths"].fillna(data["cont_child_undr_6_mnths"].median())


# In[48]:


data["is_health_worker"] = data["is_health_worker"].fillna(data["is_health_worker"].median())


# In[49]:


data.isnull().sum()


# In[50]:


data.dtypes


# # Again visualizing after cleaning the data

# In[51]:


plt.figure(figsize=(10,10))
sns.heatmap(data.isnull())


# In[52]:


data.hist(figsize=(20,15))


# ## Modelling

# In[53]:


X = data.drop(["h1n1_vaccine"], axis=1)
#dependent variable
Y= data[["h1n1_vaccine"]]


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[55]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=2)


# ### LogisticRegression

# In[56]:


lrc = LogisticRegression()
lrc.fit(X_train,Y_train)


# In[57]:


lrc.score(X_train, Y_train)


# In[58]:


lrc.score(X_test, Y_test)


# ### LogisticRegression with Standardization

# In[59]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[60]:


lrc.fit(X_train,Y_train)


# In[61]:


lrc.score(X_train, Y_train)


# In[62]:


lrc.score(X_test, Y_test)


# ### Naive bayes

# In[63]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[64]:


gnb.fit(X_train,Y_train)


# In[65]:


gnb.score(X_train,Y_train)


# In[66]:


gnb.score(X_test,Y_test)


# In[67]:


scaler_1= StandardScaler()
X_train=scaler_1.fit_transform(X_train)
X_test=scaler_1.fit_transform(X_test)


# In[68]:


gnb.fit(X_train,Y_train)


# In[69]:


gnb.score(X_train,Y_train)


# In[70]:


gnb.score(X_test,Y_test)


# In[71]:


bnb.fit(X_train,Y_train)


# In[72]:


bnb.score(X_test,Y_test)


# In[73]:


bnb.score(X_train,Y_train)


# In[74]:


scaler_2= StandardScaler()
X_train=scaler_2.fit_transform(X_train)
X_test=scaler_2.fit_transform(X_test)


# In[75]:


bnb.fit(X_train,Y_train)


# In[76]:


bnb.score(X_train,Y_train)


# In[77]:


bnb.score(X_test,Y_test)


# ### Support Vector Machine 

# In[78]:


svc =SVC(kernel='sigmoid', gamma=1.0)


# In[79]:


svc.fit(X_train,Y_train)


# In[80]:


svc.score(X_train,Y_train)


# In[81]:


svc.score(X_test,Y_test)


# ### KNeighborsClassifier

# In[82]:


knn = KNeighborsClassifier()


# In[83]:


knn.fit(X_train,Y_train)


# In[84]:


knn.score(X_train,Y_train)


# In[85]:


knn.score(X_test,Y_test)


# ### DecisionTreeClassifier

# In[91]:


dtc=DecisionTreeClassifier(max_depth=4, random_state=2)


# In[92]:


dtc.fit(X_train,Y_train)


# In[93]:


dtc.score(X_train,Y_train)


# In[94]:


dtc.score(X_test,Y_test)


# ### BaggingClassifier

# In[116]:


bc = BaggingClassifier(n_estimators=50,max_samples=1000, n_jobs=5)


# In[96]:


bc.fit(X_train,Y_train)


# In[97]:


bc.score(X_train,Y_train)


# In[98]:


bc.score(X_test,Y_test)


# ### AdaBoostClassifier 

# In[99]:


abc= AdaBoostClassifier(n_estimators=50, random_state=2)


# In[100]:


abc.fit(X_train,Y_train)


# In[101]:


abc.score(X_train,Y_train)


# In[102]:


abc.score(X_test,Y_test)


# ### RandomForestClassifier

# In[103]:


rfc= RandomForestClassifier(max_samples=1000,random_state=3)


# In[104]:


rfc.fit(X_train,Y_train)


# In[105]:


rfc.score(X_train,Y_train)


# In[106]:


rfc.score(X_test,Y_test)


# ### GradientBoostingClassifier

# In[117]:


gbdt= GradientBoostingClassifier(random_state=2,n_estimators=50)


# In[118]:


gbdt.fit(X_train,Y_train)


# In[119]:


gbdt.score(X_train,Y_train)


# In[120]:


gbdt.score(X_test,Y_test)


# ## Now finding Accuracy and Precision

# In[111]:


clfs = {
 'SVC' : svc,
    'KN' : knn,
    'BN' : bnb,
    'GB': gnb,
    'DT' : dtc,
    'LR' : lrc,
    'RF' : rfc,
    'AdaBoost' : abc,
    'BgC': bc,
    'GBDT':gbdt,
    
    
}


# In[112]:


def train_classifier(clf,X_train,X_test,Y_train,Y_test):
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    accuracy= accuracy_score(Y_test,Y_pred)
    precision= precision_score(Y_test,Y_pred)
    
    return accuracy,precision


# In[113]:


accuracy_scores= []
precision_scores=[]

for name,clf in clfs.items():
      
    current_accuracy,current_precision = train_classifier(clf,X_train,X_test,Y_train,Y_test)
        
    print('For ', name)
    print('Accuracy - ', current_accuracy)
    print('Precision - ',current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[114]:


performance_df =pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[115]:


performance_df


# # Hence Random Forest gives higher Acuuracy and precision than other algorithms

# In[ ]:




