#!/usr/bin/env python
# coding: utf-8

# <img src="mmu_logo.png" style="height: 80px;" align=left>  

# # QUESTION 3: Python Programming
# 
# Study “Open Data on COVID-19 in Malaysia” by the Ministry of Health (MOH), Malaysia via https://github.com/MoH-Malaysia/covid19-public.

# In[1]:


import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt 
from streamlit_folium import folium_static
import seaborn as sns 
import folium
import math 
from PIL import Image 

from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

from boruta import BorutaPy
from numpy import absolute
from numpy import mean
from numpy import std

Image.open('mmu_logo.png').convert('RGB').save('mmu_logo.png')
im = Image.open("mmu_logo.png")
st.image(im, width=300)

st.title("DATA MINING ASSIGNMENT")
st.header("Part (i): Discuss the exploratory data analysis steps conducted")
st.markdown("Exploratory data analysis steps included: Filling missing values with 0, Data reduction to use mostly only new cases and recovered cases, Data Discretization into weekly intervals")
# ## Part (i): Discuss the exploratory data analysis steps conducted

# Read the data dictionary/description of your dataset, as missing values sometimes are filled with dummy values.
# If you don't have access to the data dictionary, look at the descriptive statistics (mean, min, max) or make use of visualizations to try and make sense of the data. 

# ### Load Data and Filling Missing Values

# In[2]:

st.subheader("Data Cleaning")
st.markdown("Exploratory data analysis steps included: Filling missing values with 0, Data reduction to use mostly only new cases and recovered cases, Data Discretization into weekly intervals")

df_caseMsia = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv')
df_caseState = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv')
df_clusters = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/clusters.csv')
df_testMsia = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/tests_malaysia.csv')
df_testState = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/tests_state.csv')

df_caseMsia.fillna(0,inplace=True)
df_caseState.fillna(0,inplace=True)
df_clusters.fillna(0,inplace=True)
df_testMsia.fillna(0,inplace=True)
df_testState.fillna(0,inplace=True)
#df_caseMsia


# In[3]:


#df_caseState


# In[4]:


#df_clusters


# In[5]:


#df_testMsia


# In[6]:


#df_testState


#st.subheader("Data Reduction and Discretization")
# Dropped import cases and divided daily recordings into weekly ones

# In[7]:


df_caseMsia.drop(df_caseMsia.iloc[:, 4:11], inplace = True, axis = 1 )
df_caseState.drop(columns=['cases_import'], inplace = True)
df_caseMsia.drop(columns=['cases_import'], inplace = True)
date=pd.to_datetime('2021-07-01')
df_caseMsia['date'] = pd.to_datetime(df_caseMsia['date'])
df_caseMsia = df_caseMsia.resample('W-{:%a}'.format(date), on='date').sum().reset_index()


# In[8]:

st.markdown("Malaysia Cases")
df_caseMsia


# In[9]:


state = df_caseState.groupby('state')
m = state.get_group('Johor')
for name, group in state:
    date=pd.to_datetime('2020-01-25')
    group['date'] = pd.to_datetime(group['date'])
    group = group.resample('W-{:%a}'.format(date), on='date').sum().reset_index()
    group.insert(1, 'state', name)
    if name == 'Johor':
        m = group
    else:
        m = pd.concat([m, group])
df_caseState = m.sort_values('date')

st.markdown("State Cases")
df_caseState


# ### Outlier Detection
st.subheader("Outlier Detection")

# In[10]:


st.markdown("State Cases")
state = df_caseState.groupby('state')
state.boxplot(figsize=(15,15))
st.pyplot()

# ### Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")
st.markdown("Malaysia Weekly New Cases")

# In[11]:

st.altair_chart(alt.Chart(df_caseMsia).mark_line().encode(
    x='date',
    y='cases_new'
))



# State Weekly New Cases

# In[12]:

st.markdown("States Weekly New Cases")

for name, group in state:
    chart = alt.Chart(group, title=name).mark_line().encode(
    x='date',
    y=alt.Y('cases_new', scale=alt.Scale(domain=[0, 55000]))
    )
    st.altair_chart(chart)


# In[13]:


st.markdown("States Weekly New Cases Swarm Plot")
df_stateCaseNew = df_caseState.drop(columns=['cases_recovered'])
stateCN = df_stateCaseNew.groupby('state')
df_stateCaseRec = df_caseState.drop(columns=['cases_new'])
stateCR = df_stateCaseRec.groupby('state')

sns.set(rc={'figure.figsize':(11,6)})
sns.set(style="whitegrid", color_codes=True)


a = sns.swarmplot(x="state", y="cases_new", data=df_stateCaseNew)
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_title('Weekly New Cases Count Concentration per State')
a.set_ylabel('New Cases')
a.set_xlabel('States')

st.write(a)
st.pyplot()


# In[14]:


st.markdown("States Weekly Recovered Cases Swarm Plot")
sns.set(rc={'figure.figsize':(11,6)})
sns.set(style="whitegrid", color_codes=True)

a = sns.swarmplot(x="state", y="cases_recovered", data=df_stateCaseRec)
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_title('Weekly Recovered Cases Count Concentration per State')
a.set_ylabel('Recovered Cases')
a.set_xlabel('States')

st.write(a)
st.pyplot()


st.header("Part (ii): States that exhibit strong correlation with (i) Pahang, and (ii) Johor")
# ## Part (ii): States that exhibit strong correlation with (i) Pahang, and (ii) Johor

# ### New Cases

# In[15]:


st.subheader("New Cases")
m = stateCN.get_group('Pahang')
m1 = m.merge(stateCN.get_group('Johor'), left_on='date', right_on='date')
m2 = m1.merge(stateCN.get_group('Kedah'), left_on='date', right_on='date')
m3 = m2.merge(stateCN.get_group('Kelantan'), left_on='date', right_on='date')
m4 = m3.merge(stateCN.get_group('Melaka'), left_on='date', right_on='date')
m5 = m4.merge(stateCN.get_group('Negeri Sembilan'), left_on='date', right_on='date')
m6 = m5.merge(stateCN.get_group('Perak'), left_on='date', right_on='date')
m7 = m6.merge(stateCN.get_group('Perlis'), left_on='date', right_on='date')
m8 = m7.merge(stateCN.get_group('Pulau Pinang'), left_on='date', right_on='date')
m9 = m8.merge(stateCN.get_group('Sabah'), left_on='date', right_on='date')
m10 = m9.merge(stateCN.get_group('Sarawak'), left_on='date', right_on='date')
m11 = m10.merge(stateCN.get_group('Terengganu'), left_on='date', right_on='date')
m12 = m11.merge(stateCN.get_group('W.P. Kuala Lumpur'), left_on='date', right_on='date')
m13 = m12.merge(stateCN.get_group('W.P. Labuan'), left_on='date', right_on='date')
m14 = m13.merge(stateCN.get_group('W.P. Putrajaya'), left_on='date', right_on='date')
m15 = m14.merge(stateCN.get_group('Selangor'), left_on='date', right_on='date')

m15.drop(columns = ['state_x','state_y'],inplace=True)
m15.set_axis(['date', 'Pahang', 'Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan','Perak','Perlis','Pulau Pinang','Sabah','Sarawak','Terengganu','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya','Selangor'], axis='columns', inplace = True)

correlation_matrix = m15.iloc[:,:].corr()
sns.heatmap(data=correlation_matrix, annot=True)
st.pyplot()

st.markdown("Pahang most correlated state (new cases): Perak, Terengganu (0.97)")
st.markdown("Johor most correlated state (new cases): Perak, Pulau Pinang, Terengganu (0.96)")
# Pahang most correlated state: Kedah (0.97) <br>
# Johor most correlated state: Perak, Pulau Pinang, Terengganu (0.96)

# ### Recovered cases

# In[16]:


st.subheader("Recovered Cases")
m = stateCR.get_group('Pahang')
m1 = m.merge(stateCR.get_group('Johor'), left_on='date', right_on='date')
m2 = m1.merge(stateCR.get_group('Kedah'), left_on='date', right_on='date')
m3 = m2.merge(stateCR.get_group('Kelantan'), left_on='date', right_on='date')
m4 = m3.merge(stateCR.get_group('Melaka'), left_on='date', right_on='date')
m5 = m4.merge(stateCR.get_group('Negeri Sembilan'), left_on='date', right_on='date')
m6 = m5.merge(stateCR.get_group('Perak'), left_on='date', right_on='date')
m7 = m6.merge(stateCR.get_group('Perlis'), left_on='date', right_on='date')
m8 = m7.merge(stateCR.get_group('Pulau Pinang'), left_on='date', right_on='date')
m9 = m8.merge(stateCR.get_group('Sabah'), left_on='date', right_on='date')
m10 = m9.merge(stateCR.get_group('Sarawak'), left_on='date', right_on='date')
m11 = m10.merge(stateCR.get_group('Terengganu'), left_on='date', right_on='date')
m12 = m11.merge(stateCR.get_group('W.P. Kuala Lumpur'), left_on='date', right_on='date')
m13 = m12.merge(stateCR.get_group('W.P. Labuan'), left_on='date', right_on='date')
m14 = m13.merge(stateCR.get_group('W.P. Putrajaya'), left_on='date', right_on='date')
m15 = m14.merge(stateCR.get_group('Selangor'), left_on='date', right_on='date')

m15.drop(columns = ['state_x','state_y'],inplace=True)
m15.set_axis(['date', 'Pahang', 'Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan','Perak','Perlis','Pulau Pinang','Sabah','Sarawak','Terengganu','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya','Selangor'], axis='columns', inplace = True)

correlation_matrix = m15.iloc[:,:].corr()
sns.heatmap(data=correlation_matrix, annot=True)
st.pyplot()


st.markdown("Pahang most correlated state (recovered cases): Kedah (0.97)")
st.markdown("Johor most correlated state (recovered cases): Perak (0.94)")

# In[ ]:




st.header("Part (iii) Strong features/indicators to daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor")
# ## Part (iii) Strong features/indicators to daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor
st.header("(i) Pahang")
# ## Pahang

# In[17]:


df_pahang = stateCN.get_group('Pahang')
df_pahang['cases_total'] = df_pahang['cases_new'].cumsum()
df_pahang['recovered_cases'] = stateCR.get_group('Pahang')['cases_recovered']
df_pahang['kedah_newcases'] = stateCN.get_group('Kedah')['cases_new']
df_pahang['msia_newcases'] = df_caseMsia['cases_new']
df_pahang['caseState_2weeks'] = df_pahang['cases_new'].rolling(window=2).mean() 
df_pahang['caseState_4weeks'] = df_pahang['cases_new'].rolling(window=4).mean()
df_pahang['caseState_8weeks'] = df_pahang['cases_new'].rolling(window=8).mean()
df_pahang.fillna(0,inplace=True)
df_pahang

st.subheader("LASSO Regularization")
# ### LASSO Regularization

# In[18]:


reg = LassoCV()
y = df_pahang['cases_new']
X = df_pahang.drop(columns = ['cases_new','date','state'])
reg.fit(X, y)
coef = pd.Series(reg.coef_, index = X.columns)


# In[19]:




# In[20]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
st.pyplot()


st.subheader("Boruta")
# ### Boruta

# In[21]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[22]:


rf = RandomForestClassifier(n_jobs =-1, class_weight="balanced", max_depth=5)

feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)


# In[23]:


feat_selector.fit(X.values, y.values.ravel())


# In[24]:


boruta_score = ranking(list(map(float, feat_selector.ranking_)), X.columns, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values("Score",ascending = False)


# In[25]:


print('Top 10')
boruta_score


# In[26]:


sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind="bar", height=14, aspect=1.9, palette="coolwarm")
plt.title("Boruta Top 7 Features")
st.pyplot()


st.subheader("Conclusion")
st.markdown("Total cases and state new cases has the most relevance for new weekly cases for Pahang")

st.header("(ii) Kedah")
# ## Kedah

# In[27]:


df_kedah = stateCN.get_group('Kedah')
df_kedah['cases_total'] = df_kedah['cases_new'].cumsum()
df_kedah['recovered_cases'] = stateCR.get_group('Kedah')['cases_recovered']
df_kedah['msia_newcases'] = df_caseMsia['cases_new']
df_kedah['caseState_2weeks'] = df_kedah['cases_new'].rolling(window=2).mean() 
df_kedah['caseState_4weeks'] = df_kedah['cases_new'].rolling(window=4).mean()
df_kedah['caseState_8weeks'] = df_kedah['cases_new'].rolling(window=8).mean()
df_kedah.fillna(0,inplace=True)
df_kedah


st.subheader("LASSO Regularization")
# ### LASSO Regularization

# In[28]:


reg = LassoCV()
y = df_kedah['cases_new']
X = df_kedah.drop(columns = ['cases_new','date','state'])
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[29]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[30]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
st.pyplot()


st.subheader("Boruta")
# ### Boruta

# In[31]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[32]:


rf = RandomForestClassifier(n_jobs =-1, class_weight="balanced", max_depth=5)

feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)


# In[33]:


feat_selector.fit(X.values, y.values.ravel())


# In[34]:


boruta_score = ranking(list(map(float, feat_selector.ranking_)), X.columns, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values("Score",ascending = False)


# In[35]:


print('Top 10')
boruta_score


# In[36]:


sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind="bar", height=14, aspect=1.9, palette="coolwarm")
plt.title("Boruta Top 7 Features")
st.pyplot()
st.subheader("Conclusion")
st.markdown("Malaysia new cases has the most relevance for new weekly cases for Kedah")


st.header("(iii) Johor")
# ## Johor

# In[37]:


df_johor = stateCN.get_group('Johor')
df_johor['cases_total'] = df_johor['cases_new'].cumsum()
df_johor['recovered_cases'] = stateCR.get_group('Johor')['cases_recovered']
df_johor['kedah_newcases'] = stateCN.get_group('Perak')['cases_new']
df_johor['msia_newcases'] = df_caseMsia['cases_new']
df_johor['caseState_2weeks'] = df_johor['cases_new'].rolling(window=2).mean() 
df_johor['caseState_4weeks'] = df_johor['cases_new'].rolling(window=4).mean()
df_johor['caseState_8weeks'] = df_johor['cases_new'].rolling(window=8).mean()
df_johor.fillna(0,inplace=True)
df_johor


st.subheader("LASSO Regularization")
# ### LASSO Regularization

# In[38]:


reg = LassoCV()
y = df_johor['cases_new']
X = df_johor.drop(columns = ['cases_new','date','state'])
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[39]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[40]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
st.pyplot()


st.subheader("Boruta")
# ### Boruta

# In[41]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[42]:


rf = RandomForestClassifier(n_jobs =-1, class_weight="balanced", max_depth=5)

feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)


# In[43]:


feat_selector.fit(X.values, y.values.ravel())


# In[44]:


boruta_score = ranking(list(map(float, feat_selector.ranking_)), X.columns, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values("Score",ascending = False)


# In[45]:


print('Top 10')
boruta_score


# In[46]:


sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind="bar", height=14, aspect=1.9, palette="coolwarm")
plt.title("Boruta Top 7 Features")
st.pyplot()


st.subheader("Conclusion")
st.markdown("Malaysia new cases has the most relevance for new weekly cases for Johor")

st.header("(iv) Selangor")
# ## Selangor

# In[47]:


df_selangor = stateCN.get_group('Selangor')
df_selangor['cases_total'] = df_selangor['cases_new'].cumsum()
df_selangor['recovered_cases'] = stateCR.get_group('Selangor')['cases_recovered']
df_selangor['msia_newcases'] = df_caseMsia['cases_new']
df_selangor['caseState_2weeks'] = df_selangor['cases_new'].rolling(window=2).mean() 
df_selangor['caseState_4weeks'] = df_selangor['cases_new'].rolling(window=4).mean()
df_selangor['caseState_8weeks'] = df_selangor['cases_new'].rolling(window=8).mean()
df_selangor.fillna(0,inplace=True)
df_selangor


st.subheader("LASSO Regularization")
# ### LASSO Regularization

# In[48]:


reg = LassoCV()
y = df_selangor['cases_new']
X = df_selangor.drop(columns = ['cases_new','date','state'])
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)


# In[49]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[50]:


imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
st.pyplot()


st.subheader("Boruta")
# ### Boruta

# In[51]:


def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


# In[52]:


rf = RandomForestClassifier(n_jobs =-1, class_weight="balanced", max_depth=5)

feat_selector = BorutaPy(rf, n_estimators="auto",random_state=1)


# In[53]:


feat_selector.fit(X.values, y.values.ravel())


# In[54]:


boruta_score = ranking(list(map(float, feat_selector.ranking_)), X.columns, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
boruta_score = boruta_score.sort_values("Score",ascending = False)


# In[55]:


print('Top 10')
boruta_score


# In[56]:


sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind="bar", height=14, aspect=1.9, palette="coolwarm")
plt.title("Boruta Top 7 Features")
st.pyplot()

st.subheader("Conclusion")
st.markdown("Malaysia new cases and State Cases 2 weeks has the most relevance for new weekly cases for Selangor")

st.header("Part (iv): Model performs well in predicting the daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor")
# ## Part (iv): Model performs well in predicting the daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor

st.header("(i) Pahang")
# ## Pahang
st.subheader("Lasso Regression")
# ### Lasso Regression

# In[57]:


model = Lasso()
y = df_pahang['cases_new']
X = df_pahang.drop(columns = ['cases_new','date','state'])
labels = ['low','medium','high']
df_pahang['bins'] = pd.cut(df_pahang['cases_new'], bins = [-np.inf, np.percentile(df_pahang['cases_new'],33), np.percentile(df_pahang['cases_new'],67), df_pahang['cases_new'].max()], labels=labels)
df_pahang['bins'].value_counts()
X = df_pahang.drop(columns = ['caseState_8weeks','caseState_4weeks','caseState_2weeks', 'recovered_cases','date','state', 'cases_new', 'bins'])
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=5)


# In[58]:


model.fit(X_train, y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predictL = model.predict(X_test)
#coeff_used = np.sum(model.coef_!=0)

# print ("training score:", train_score)
# print ("test score: ", test_score)
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictL))))
print(math.sqrt(mean_squared_error(y_test, y_predictL)))
#print ("number of features used: ", coeff_used)


st.subheader("Polynomial Regression")
# ### Polynomial Regression

# In[59]:


poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
y_predictP = lin2.predict(poly.fit_transform(X_test))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictP))))


# In[60]:


fig, axes = plt.subplots(figsize=(15,5), dpi=100)
plt.plot([n for n in range(len(X_test))], y_test, label='Actual')
plt.plot([n for n in range(len(X_test))], y_predictL, label='Lasso Regression')
plt.plot([n for n in range(len(X_test))], y_predictP, label='Polynomial Regression')
plt.xlabel('Index')
plt.ylabel('New Cases')
plt.grid()
plt.legend()
plt.show()
st.pyplot()

st.subheader("Conclusion")
st.markdown("Polynomial Regression is the more appropriate regression method for predicting")

st.subheader("Gradient Boosting Classification")
# ### Gradient Boosting Classification

# In[61]:


y = df_pahang['bins']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.fit(X_train,y_train)
y_pred = gradient_booster.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol

st.subheader("Random Forest Classifier")
# ### Random Forest Classifier

# In[62]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol

st.subheader("Conclusion")
st.markdown("Random Forest Classifier is the more appropriate classificaiton method for predicting")

st.header("(ii) Kedah")
# ## Kedah

st.subheader("Lasso Regression")
# ### Lasso Regression

# In[63]:


model = Lasso()
y = df_kedah['cases_new']
X = df_kedah.drop(columns = ['cases_new','date','state'])
labels = ['low','medium','high']
df_kedah['bins'] = pd.cut(df_kedah['cases_new'], bins = [-np.inf, np.percentile(df_kedah['cases_new'],33), np.percentile(df_kedah['cases_new'],67), df_kedah['cases_new'].max()], labels=labels)
df_kedah['bins'].value_counts()
X = df_kedah.drop(columns = ['caseState_8weeks','caseState_4weeks','caseState_2weeks', 'recovered_cases','date','state', 'cases_new', 'bins', 'cases_total'])
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=5)


# In[64]:


model.fit(X_train, y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predictL = model.predict(X_test)
#coeff_used = np.sum(model.coef_!=0)

# print ("training score:", train_score)
# print ("test score: ", test_score)
print(math.sqrt(mean_squared_error(y_test, y_predictL)))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictL))))
#print ("number of features used: ", coeff_used)


st.subheader("Polynomial Regression")
# ### Polynomial Regression

# In[65]:


poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
y_predictP = lin2.predict(poly.fit_transform(X_test))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictP))))


# In[66]:


fig, axes = plt.subplots(figsize=(15,5), dpi=100)
plt.plot([n for n in range(len(X_test))], y_test, label='Actual')
plt.plot([n for n in range(len(X_test))], y_predictL, label='Lasso Regression')
plt.plot([n for n in range(len(X_test))], y_predictP, label='Polynomial Regression')
plt.xlabel('Index')
plt.ylabel('New Cases')
plt.grid()
plt.legend()
plt.show()
st.pyplot()

st.subheader("Conclusion")
st.markdown("Polynomial Regression is the more appropriate regression method for predicting")

st.subheader("Gradient Boosting Classification")
# ### Gradient Boosting Classification

# In[67]:


y = df_kedah['bins']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.fit(X_train,y_train)
y_pred = gradient_booster.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol


st.subheader("Random Forest Classifier")
# ### 

# In[68]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol

st.subheader("Conclusion")
st.markdown("Oddly enough, both methods, Gradient Boosting and Random Forest are equal classificaiton methods for predicting")

st.header("(iii) Johor")
# ## Johor

st.subheader("Lasso Regression")
# ### Lasso Regression

# In[69]:


model = Lasso()
y = df_johor['cases_new']
X = df_johor.drop(columns = ['cases_new','date','state'])
labels = ['low','medium','high']
df_johor['bins'] = pd.cut(df_johor['cases_new'], bins = [-np.inf, np.percentile(df_johor['cases_new'],33), np.percentile(df_johor['cases_new'],67), df_johor['cases_new'].max()], labels=labels)
df_johor['bins'].value_counts()
X = df_johor.drop(columns = ['caseState_8weeks','caseState_4weeks','caseState_2weeks', 'recovered_cases','date','state', 'cases_new', 'bins', 'cases_total'])
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=5)


# In[70]:


model.fit(X_train, y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predictL = model.predict(X_test)
#coeff_used = np.sum(model.coef_!=0)

# print ("training score:", train_score)
# print ("test score: ", test_score)
print(math.sqrt(mean_squared_error(y_test, y_predictL)))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictL))))
#print ("number of features used: ", coeff_used)


st.subheader("Polynomial Regression")
# ### Polynomial Regression

# In[71]:


poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
y_predictP = lin2.predict(poly.fit_transform(X_test))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictP))))


# In[72]:


fig, axes = plt.subplots(figsize=(15,5), dpi=100)
plt.plot([n for n in range(len(X_test))], y_test, label='Actual')
plt.plot([n for n in range(len(X_test))], y_predictL, label='Lasso Regression')
plt.plot([n for n in range(len(X_test))], y_predictP, label='Polynomial Regression')
plt.xlabel('Index')
plt.ylabel('New Cases')
plt.grid()
plt.legend()
plt.show()
st.pyplot()

st.subheader("Conclusion")
st.markdown("Polynomial Regression is the more appropriate regression method for predicting")

st.subheader("Gradient Boosting Classification")
# ### Gradient Boosting Classification

# In[73]:


y = df_johor['bins']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.fit(X_train,y_train)
y_pred = gradient_booster.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol


st.subheader("Random Forest Classifier")
# ### Random Forest Classifier

# In[74]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol

st.subheader("Conclusion")
st.markdown("Random Forest is the more appropriate classificaiton method for predicting")

st.header("(iv) Selangor")
# ## Selangor

st.subheader("Lasso Regression")
# ### Lasso Regression

# In[75]:


model = Lasso()
y = df_selangor['cases_new']
X = df_selangor.drop(columns = ['cases_new','date','state'])
labels = ['low','medium','high']
df_selangor['bins'] = pd.cut(df_selangor['cases_new'], bins = [-np.inf, np.percentile(df_selangor['cases_new'],33), np.percentile(df_selangor['cases_new'],67), df_selangor['cases_new'].max()], labels=labels)
df_selangor['bins'].value_counts()
X = df_selangor.drop(columns = ['caseState_8weeks','caseState_4weeks', 'recovered_cases','date','state', 'cases_new', 'bins', 'msia_newcases', 'cases_total'])
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=5)


# In[76]:


model.fit(X_train, y_train)
train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)
y_predictL = model.predict(X_test)
#coeff_used = np.sum(model.coef_!=0)

# print ("training score:", train_score)
# print ("test score: ", test_score)
print(math.sqrt(mean_squared_error(y_test, y_predictL)))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictL))))
#print ("number of features used: ", coeff_used)


st.subheader("Polynomial Regression")
# ### Polynomial Regression

# In[77]:


poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)
  
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
y_predictP = lin2.predict(poly.fit_transform(X_test))
st.markdown("Mean Squared Error: " + str(math.sqrt(mean_squared_error(y_test, y_predictP))))


# In[78]:


fig, axes = plt.subplots(figsize=(15,5), dpi=100)
plt.plot([n for n in range(len(X_test))], y_test, label='Actual')
plt.plot([n for n in range(len(X_test))], y_predictL, label='Lasso Regression')
plt.plot([n for n in range(len(X_test))], y_predictP, label='Polynomial Regression')
plt.xlabel('Index')
plt.ylabel('New Cases')
plt.grid()
plt.legend()
plt.show()
st.pyplot()

st.subheader("Conclusion")
st.markdown("Polynomial Regression is the more appropriate regression method for predicting")

st.subheader("Gradient Boosting Classification")
# ### Gradient Boosting Classification

# In[79]:


y = df_selangor['bins']
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.fit(X_train,y_train)
y_pred = gradient_booster.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol


st.subheader("Random Forest Classifier")
# ### Random Forest Classifier

# In[80]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=2)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('score: ', accuracy_score(y_test, y_pred))
st.markdown('score: ' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))
lol = classification_report(y_test,y_pred)
lol


st.subheader("Conclusion")
st.markdown("Gradient Boosting Classifier is the more appropriate classificaiton method for predicting")