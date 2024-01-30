#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ### Load dataset

# In[63]:


df = pd.read_csv(r"C:\Users\Kim Yuri\Desktop\Projects\R projects\Titanic dataset\tested.csv")
df


# ### Dataset notes

# Embarked - Where the traveler mounted from (Queenstown, Cherbourg, Southampton)
# Parch - Number of parents/children aboard
# Pclass - Passenger class (1 =1st, 2 = 2nd, 3 = 3rd)
# Survival - (0 = No, 1 = Yes)
# sibsp - Number of siblings/spouses aboard
# fare - British pound passenger fare
# 
# 
# * To do list next:
# Replace the NA value on Fare (Determine the average fare for a certain cabin or class)
# 
# Then proceed with replacing NA values on Age

# ### Questions from Chat GPT 
# Certainly! Here are some questions you can explore using the Titanic dataset for your Exploratory Data Analysis (EDA) project:
# 
# What is the distribution of passenger ages on the Titanic?
# 
# How many passengers survived, and how many did not survive?
# 
# What is the distribution of passenger classes (1st, 2nd, 3rd)?
# 
# Are there any correlations between age and survival?
# 
# What is the gender distribution among passengers?
# 
# How does the distribution of fares look like?
# 
# Are there any relationships between the passenger class and survival?
# 
# What is the distribution of embarkation locations?
# 
# Is there any correlation between the number of siblings/spouses aboard and survival?
# 
# How many passengers were traveling with parents or children, and how did it affect survival?

# In[64]:


df.info()


# ### Replace Embarked location with full location name

# In[67]:


embark_loc = {'Q': 'Queenstown', 'C': 'Cherbourg', 'S': 'Southampton'}


# In[65]:


df1 = df.copy()


# In[66]:


df1


# In[68]:


df1['Embarked'] = df1['Embarked'].map(embark_loc)
df1.head(5)


# #### What is the distribution of passenger ages on the Titanic?

# In[69]:


plt.hist(df1['Age'], edgecolor = 'black', bins=10)
plt.title('Distribution of age in Titanic')
plt.show()


# #### How many passengers survived, and how many did not survive?

# In[7]:


survive = {0:'No', 1:'Yes'}


# In[70]:


df1['Survived'] = df1['Survived'].map(survive)
df1.head(5)


# In[71]:


survived_count = df1['Survived'].value_counts()
survived_count = pd.DataFrame(survived_count)
survived_count


# In[72]:


survived_count.plot(kind='pie',subplots=True, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], legend=True)
# pie charts require y or subplots=True 
plt.title('Distribution of those who survived in Titanic')


# #### What is the distribution of passenger classes (1st, 2nd, 3rd)?

# In[73]:


classes = {3: 'Third class', 2:'Second class', 1:'First class'}


# In[74]:


df1['Pclass'] = df1['Pclass'].map(classes)
df1


# In[75]:


pclass = df1['Pclass'].value_counts()
pclass = pd.DataFrame(pclass)
pclass


# In[76]:


pclass.plot(kind='pie',subplots=True, autopct='%1.1f%%', colors=['skyblue', 'orange','green'], legend=False)
# pie charts require y or subplots=True 
plt.title('Distribution of passenger classes in Titanic')


# #### Are there any correlations between age and survival?

# In[77]:


age_survival = ['Age', 'Survived']


# In[78]:


age_survival_df = df1[age_survival]
age_survival_df


# In[79]:


reverse_survive = {'No':0, 'Yes':1}


# In[80]:


age_survival_df.loc[:, 'Survived'] = age_survival_df['Survived'].map(reverse_survive)
age_survival_df

#use .loc instead, first argument :, means we want to get entire rows, then second argument is the column 'Survived'


# In[81]:


age_survival_corr = age_survival_df.corr()
age_survival_corr


# In[82]:


sns.heatmap(age_survival_corr, annot=True)

# There is a negative correlation between Age and Survival


# #### What is the gender distribution among passengers?

# In[83]:


gender = pd.DataFrame(df1['Sex'].value_counts())
gender


# In[84]:


gender.plot(kind='pie',subplots=True, autopct='%1.1f%%', colors=['skyblue', 'pink'], legend=False)
plt.title('Distribution of gender of passengers in Titanic')
plt.show()


# #### How does the distribution of fares look like?

# In[85]:


df1.info()


# In[86]:


df1.describe()


# In[87]:


average_fare_class = df1.groupby('Pclass')['Fare'].mean()
average_fare_class


# In[88]:


average_fare_class.plot(kind='bar', color='green')
plt.title('Average fare across boarding class in Titanic')
plt.show()


# In[89]:


average_fare_parch = df1.groupby('Parch')['Fare'].mean()
average_fare_parch


# In[90]:


average_fare_parch.plot(kind='bar', color='green')
plt.title('Average fare per number of Parents/Children onboarded in Titanic')
plt.show()


# In[91]:


df1['Parch'].value_counts()

# Parch to Fare is not a really good basis for comparing Fare


# In[92]:


df1['SibSp'].value_counts()


# In[93]:


average_fare_sibsp = df1.groupby('SibSp')['Fare'].mean()
average_fare_sibsp

# most likely not a good indicator of fare as well since the max fare did not have the corresponding max sibs/spouse 


# In[94]:


df


# #### Are there any relationships between the passenger class and survival?

# In[120]:


sns.countplot(x = 'Pclass', hue = 'Survived', data = df1)


# In[133]:


pclass_survived = df[['Pclass','Survived']]
pclass_survived


# In[148]:


rng = np.random.default_rng(seed=42)
colors = rng.random(len(df))


# In[150]:


plt.scatter(df['Pclass'], df['Survived'], c= colors, alpha=1.0)
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.title('Scatter Plot with Random Colors')


# In[151]:


pclass_survived.corr() 
# low negative correlation between Pclass and if the passenger survived or not


# #### What is the distribution of embarkation locations?

# In[170]:


df1.groupby('Embarked').size()


# In[171]:


embarked_pclass = df1.groupby(['Embarked', 'Pclass']).size()
embarked_pclass 


# In[172]:


embarked_pclass = embarked_pclass.unstack()


# In[173]:


embarked_pclass_df = pd.DataFrame(embarked_pclass)
embarked_pclass_df


# In[178]:


embarked_pclass_df = embarked_pclass_df.reset_index()
embarked_pclass_df 


# In[175]:


melt_embarked_pclass_df = pd.melt(embarked_pclass_df, id_vars=['Embarked'], value_vars=['First class', 'Second class', 'Third class'], var_name='Pclass', value_name='Count')


# In[179]:


melt_embarked_pclass_df


# In[182]:


sns.barplot(x = 'Embarked', y = 'Count', hue = 'Pclass', data = melt_embarked_pclass_df, palette = 'viridis')


# sns.barplot is best suitable to get the central tendency of multiple groups
# requires x and y where x is categorical variable and y is numerical variable
# 
# 
# sns.countplot is best suitable to visualize the count/frequencies for each categories 
# used for comparing the distribution of categorical variables

# #### Is there any correlation between the number of siblings/spouses aboard and survival?

# In[197]:


sibsp_survival = df1.groupby(['SibSp', 'Survived']).size()
sibsp_survival


# In[198]:


sibsp_survival_df = pd.DataFrame(sibsp_survival.unstack())


# In[199]:


sibsp_survival_df


# In[200]:


sibsp_survival_df.fillna(0, inplace=True)


# In[201]:


sibsp_survival_df
# reset_index then melt to transform the dataframe into something easy to visualize


# In[203]:


sibsp_survival_df = sibsp_survival_df.reset_index()
sibsp_survival_df


# In[204]:


melt_sibsp_survival_df = pd.melt(sibsp_survival_df, id_vars=['SibSp'], var_name='Survived', value_name='Count')


# In[205]:


melt_sibsp_survival_df


# In[206]:


sns.barplot(x='SibSp', y='Count', hue='Survived', data=melt_sibsp_survival_df)


# #### How many passengers were traveling with parents or children, and how did it affect survival?

# In[223]:


df1.groupby(['Parch','Survived']).size()


# In[225]:


parch_survived_df = df1.groupby(['Parch','Survived']).size()


# In[228]:


parch_survived_df = parch_survived_df.reset_index()
parch_survived_df


# In[229]:


parch_survived_df = pd.DataFrame(parch_survived_df)
parch_survived_df


# In[230]:


sns.barplot(x = 'Parch', y = 0, hue = 'Survived', data = parch_survived_df)
plt.xlabel('Number of children and parents onboarded')
plt.ylabel('Count')


# In[ ]:




