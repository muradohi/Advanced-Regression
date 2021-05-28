#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#shows all the columns
pd.set_option('display.max_columns',None)


# In[2]:


dataset = pd.read_csv('train.csv')


# In[3]:


dataset


# In[ ]:





# # MISSING VALUES

# In[4]:


dataset.columns


# In[ ]:





# In[5]:


features_nan = [features for features in dataset.columns if dataset[features].isnull().sum()>1]
for features in features_nan:
    
    print(features,':', np.round(dataset[features].isnull().mean()*100,4), '% missing values',',','Total',':' ,len(dataset[features]))


# In[6]:


features_nan


# In[7]:


dataset[features_nan]


# # Relation between nan values and Sales Price

# In[8]:


for feature in features_nan:
    
    data = dataset.copy()
    #data[feature]=np.where(data[feature].isnull(),0,1)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.legend()
    plt.show()
    


# In[9]:


for feature in features_nan:
    
    data = dataset.copy()
    data[feature]=np.where(data[feature].isnull(),0,1)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.legend()
    plt.show()


# # Numerical Variables

# In[10]:


num_feature = [feat for feat in dataset.columns if dataset[feat].dtype!='O']
dataset[num_feature]
    


# In[11]:


num_feature


# ######year in numerical features

# In[12]:


year_feature =[feat for feat in num_feature if 'Yr' in feat or 'Year' in feat] 


# In[13]:


year_feature


# In[14]:


dataset[year_feature]


# In[15]:


dataset.groupby('YrSold')['SalePrice'].median().plot()


# In[16]:


for f in year_feature:
    if f != 'YrSold':
        
        data = dataset.copy()
        data[f] = data['YrSold'] - data[f]
        plt.scatter(data[f],data['SalePrice'])
        plt.title(f)
        plt.show()
        
        


# In[17]:


discrete_feature = [feat for feat in num_feature if len(dataset[feat].unique())<=25 and feat not in year_feature+['Id']]

for feat in discrete_feature:
    print(feat,',', 'Unique :', len(dataset[feat].unique()))


# In[18]:


dataset[discrete_feature]


# In[19]:


dataset['MiscVal'].value_counts()


# In[20]:


plt.scatter(dataset['MiscVal'],dataset['SalePrice'])


# In[21]:


for feat in discrete_feature:
    data = dataset.copy()
    data.groupby(feat)['SalePrice'].median().plot.bar()
    plt.xlabel(feat)
    plt.ylabel('SalePrice')
    
    plt.show()
    


# In[22]:


continuous_feature = [feat for feat in num_feature if feat not in discrete_feature+year_feature+['Id']]


# In[23]:


continuous_feature


# In[24]:


dataset['LotFrontage'].value_counts()


# In[25]:


dataset[continuous_feature]


# In[26]:


for feat in continuous_feature:
    print(feat,',','Unique :',len(dataset[feat].unique()))


# In[ ]:





# In[27]:


for feat in continuous_feature:
    dataset[feat].hist(bins=30)
    plt.xlabel(feat)
    plt.ylabel('SalePrice')
    plt.show()


# In[28]:


for feat in continuous_feature:
    data = dataset.copy()
    plt.scatter(data[feat],data['SalePrice'])
    plt.xlabel(feat)
    plt.ylabel('SalePrice')
    plt.show()


# In[29]:


for feat in continuous_feature:
    data = dataset.copy()
    data[feat] = np.log(data[feat])
    data['SalePrice'] = np.log(data['SalePrice'])
    plt.scatter(data[feat],data['SalePrice'])
    
    plt.xlabel(feat)
    plt.ylabel('SalePrice')
    plt.show()


# In[30]:


for feat in continuous_feature:
    data = dataset.copy()
    if 0 in data[feat].unique():
        pass
    else:
        
        data[feat] = np.log(data[feat])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feat],data['SalePrice'])
        plt.xlabel(feat)
        plt.ylabel('SalePrice')
        plt.show()
        


# In[ ]:





# # Outliers

# In[31]:


for feat in continuous_feature:
    data = dataset.copy()
    if 0 in data[feat].unique():
        pass
    else:
        data.boxplot(column=feat)
        
        
        plt.show()


# In[32]:


for feat in continuous_feature:
    data = dataset.copy()
    unique = data[feat].unique()
    print(feat , unique)


# In[33]:


for feat in continuous_feature:
    data = dataset.copy()
    if 0  in data[feat].unique():
        unique = data[feat].unique()
        print(feat , unique)


# In[34]:


dataset['BsmtFinSF1'].unique() ==0


# In[35]:


categorical_features = [feat for feat in dataset.columns if dataset[feat].dtype=='O']


# In[36]:


dataset[categorical_features]


# In[37]:


for feat in categorical_features:
    data = dataset.copy()
    
    data.groupby([feat])['SalePrice'].median().plot.bar()
    plt.show()


# In[ ]:





# # End of data analysis

# In[ ]:





# #  Feature Engineering

# MISSING VALUES

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X = dataset


# In[40]:


y = dataset['SalePrice']


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)


# In[42]:


X_train


# In[43]:


y_train


# In[ ]:





# In[44]:


num_nan_feat = [feat for feat in dataset.columns if dataset[feat].isnull().sum()>1 and feat in num_feature and feat not in year_feature]


# In[45]:


num_nan_feat


# In[46]:


cate_nan_feat = [feat for feat in dataset.columns if dataset[feat].isnull().sum()>1 and feat in categorical_features and feat not in year_feature]


# In[47]:


len(cate_nan_feat)


# In[48]:


cate_nan_feat


# In[49]:


dataset[cate_nan_feat]


# In[50]:


dataset.isnull().sum().head(10)


# In[51]:


def impute_cate_nan(dataset,cate_nan_feat):
    df = dataset.copy()
    df[cate_nan_feat] = df[cate_nan_feat].fillna('Missing')
    return df
    


# In[52]:


dataset = impute_cate_nan(dataset,cate_nan_feat)


# In[53]:


dataset


# In[54]:


df = dataset.copy()


# In[55]:


df


# In[56]:


for feat in num_nan_feat:
    median = X_train[feat].median()

    X_train[feat] = X_train[feat].fillna(median)
    


# In[57]:


for feat in num_nan_feat:
    median = X_test[feat].median()

    X_test[feat] = X_test[feat].fillna(median)


# In[58]:


X_test['LotFrontage'].isnull().sum()


# In[59]:


dataset = pd.concat([X_train,X_test]).sort_index()


# In[60]:


dataset['LotFrontage'].isnull().sum()


# In[61]:


dataset


# In[62]:


df1 = dataset.copy()


# In[63]:


dataset[year_feature].isnull().sum()


# In[64]:


dataset['GarageYrBlt'].value_counts()


# In[65]:


dataset['GarageYrBlt'].median()


# In[66]:


dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].median())


# In[67]:


dataset[year_feature].isnull().sum()


# In[68]:


dataset[categorical_features].isnull().sum()


# In[69]:


dataset = impute_cate_nan(dataset,cate_nan_feat)


# In[70]:


dataset[categorical_features].isnull().sum()


# In[71]:


len(categorical_features)+len(num_feature)


# # Some numerical variavles are skwed  

# # Perform Log nornal distribution

# In[72]:


num_feat_skwed = ['LotFrontage','GrLivArea','1stFlrSF','SalePrice','LotArea']


# In[73]:


for feat in num_feat_skwed:
    dataset[feat] = np.log(dataset[feat])


# In[74]:


dataset


# In[75]:


for feat in year_feature:
    if feat == 'YrSold':
        pass
    else:
        dataset[feat] = dataset['YrSold']-dataset[feat]


# In[76]:


dataset[year_feature]


# In[77]:


dataset


# In[78]:


df2 = dataset.copy()


# In[79]:


df2.head(20)


# In[80]:


for feat in categorical_features:
    label = df2.groupby(feat)['SalePrice'].mean().sort_values().index
    label = {k:i for i,k in enumerate(label,0)}
    df2[feat] = df2[feat].map(label)


# In[81]:


df2


# In[82]:


dataset = df2.copy()


# In[83]:


dataset


# In[84]:


#df2.groupby('LotConfig')['SalePrice'].mean().sort_values().index


# In[85]:


#label_LotConfig = df2.groupby('LotConfig')['SalePrice'].mean().sort_values().index
#for k,i  in enumerate(label,0):
    #print({i,k})


# In[ ]:





# In[86]:


scaling_feat = [feat for feat in dataset.columns if feat not in ['Id','SalePrice'] ]


# In[87]:


scaling_feat


# In[91]:


from sklearn.preprocessing import MinMaxScaler


# In[92]:


scaler = MinMaxScaler()


# In[93]:


scaler.fit_transform(dataset[scaling_feat])


# In[94]:


scaled_dataset = pd.DataFrame(scaler.fit_transform(dataset[scaling_feat]),columns=scaling_feat)


# In[95]:


scaled_dataset


# In[96]:


data = pd.concat([dataset[['Id','SalePrice']].reset_index(drop=True),scaled_dataset],axis= 1)


# In[97]:


data


# In[ ]:





# In[99]:


sns.heatmap(data.isnull())


# In[100]:


data.columns


# In[101]:


data['Electrical'].isnull().sum()


# In[102]:


data = data.dropna()


# In[103]:


data


# In[104]:


data.to_csv('House_data_scaled',index=False)


# In[ ]:




