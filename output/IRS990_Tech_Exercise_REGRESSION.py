
# coding: utf-8

# # Data Insights Take-Home: Basic Regression
# **Marianne C. Halloran
# October 12, 2017**
# 
# *Simple regression analisys showing the relationship between Net Assets, Total Expenses and Total Revenue*

# In[2]:

from __future__ import print_function
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from pylab import rcParams
# get_ipython().magic(u'matplotlib inline')
rcParams['figure.figsize'] = 10,10
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D


# In[4]:

#=================================================================#
# DATA IMPPORT                                                    #
#=================================================================#
meta = pd.read_csv('../input/NPO_meta_38k.csv')
meta.columns = ['EIN','contract_term','tax_status','org_name','city','state','tax_year',
                'activity','year_formed','volunteer_ct','employee_ct','rev_campaigns',
                'rev_membership', 'rev_fundraising','rev_govgrants','rev_other',
                'rev_progserv','rev_netfundraising','total_revenue','total_revenuePY',
                'exp_grants','exp_progserv', 'exp_management','exp_fundraising','total_expenses',
                'total_compensations','comp_more100k', 'net_assets','pol_act','lob_act',
                'foreign_office','foreign_fundraising','foreign_assist']
del meta['EIN'],  meta['contract_term']# meta['activity'],meta['year_formed'],
print(u"\u0011",'Clean data, removed NaN')


# I'm removing any organization that is not a 501(c)(3) and any orgs with NaN in a row
meta = meta.dropna(axis=0,how='any')
meta_501c3 = meta.loc[meta['tax_status'] == 0]
del meta; meta = meta_501c3
meta


# In[5]:

#=================================================================#
# DESCRIPTIVE STATISTICS                                          #
#=================================================================#
print(u"\u0011",'Descriptive statistics, summarizing central tendency, dispersion')
print('  and shape of dataset\'s distribution')
meta.describe()


# In[6]:

#=================================================================#
# PROCESS DATA: Categorical conversions, OHE, features            #
#=================================================================#
# Cities and States will get categorical codes
meta['city'] = meta['city'].str.upper() # all upper case
cities = sorted(meta['city'].unique())  # sort by unique names
meta['city_int'] = meta['city'].map(lambda x: cities.index(x))
# states = sorted(meta['state'].unique())
# meta['state_int'] = meta['state'].map(lambda x: states.index(x))
meta


# In[7]:

#=================================================================#
# LOGISTIC REGRESSION                                             #
#=================================================================#
## 
# Standarize (z-score) array (zi = xi-xmean/std)
meta1 = (meta[['total_expenses', 'total_revenue']].copy()).apply(stats.zscore)
meta1_z = meta1[(np.abs(stats.zscore(meta1)) < 3).all(axis=1)]

# Visualization option
keys = meta1_z.index.get_values()
net_assets = meta['net_assets'].loc[keys]
volume = (10+ net_assets/3000000)

## Visualizations
plt.scatter(meta1['total_expenses'], meta1['total_revenue'], s=50);
plt.title('Scatter plot of Total Expenses vs Total Revenue', fontsize=16)
plt.xlabel('Total Expenses(U$)'); plt.ylabel('Total Revenue (U$)'); plt.show()

plt.scatter(meta1_z['total_expenses'], meta1_z['total_revenue'], s=volume);
plt.title('[Z-score Normalized, No Outliers] Scatter plot of Total Expenses vs Total Revenue,\nSize=net_asset', fontsize=16)
plt.xlabel('z(Total Expenses)'); plt.ylabel('z(Total Revenue)'); plt.show()
plt.show()


# In[10]:

#=================================================================#
# PROCESS DATA: Categorical conversions, OHE, features            #
#=================================================================#
keys = meta1_z.index.get_values()
meta1_z = meta1_z.assign(city_int = meta['city_int'].loc[keys],
#                          state_int = meta['state_int'].loc[keys],
                         net_assets = meta['net_assets'].loc[keys])

# 3D Plot vs Cities
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(meta1_z['net_assets'],meta1_z['total_revenue'],meta1_z['total_expenses'], s=volume)
ax.set_zlabel('Total Expenses (US$)')
ax.set_ylabel('Total Revenue( US$)')
ax.set_xlabel('Net Assets (US$)')
ax.set_title('Net Assets, Total Expenses and Total Revenue', fontsize=16)
ax.view_init(elev=20., azim=60)
plt.show()


# In[11]:

#=================================================================#
# LINEAR REGRESSION:   Y = a.X1 + b.X2 + c                        #
#=================================================================#
# OLS method of statsmodels 
# one response and two predictor variables
print(u"\u0011","LR Model Fitting Results")
model = smf.ols(formula='net_assets ~ total_expenses + total_revenue', data=meta1_z)
results_formula = model.fit()
results_formula.params


# In[12]:

x_surf, y_surf = np.meshgrid(np.linspace(meta1_z.total_expenses.min(), 
                                         meta1_z.total_expenses.max(), 100),np.linspace(meta1_z.total_revenue.min(), meta1_z.total_revenue.max(), 10))
onlyX = pd.DataFrame({'total_expenses': x_surf.ravel(), 'total_revenue': y_surf.ravel()})
fittedY=results_formula.predict(exog=onlyX)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(meta1_z['total_expenses'],meta1_z['total_revenue'],meta1_z['net_assets'],s=volume, c='blue', marker='o', alpha=0.8)
ax.plot_surface(x_surf,y_surf,fittedY.values.reshape(x_surf.shape), color='black', alpha=.8)
ax.view_init(elev=20., azim=160)
ax.set_xlabel('Total Expenses')
ax.set_ylabel('Total Revenue')
ax.set_zlabel('Net Assets')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(meta1_z['total_expenses'],meta1_z['total_revenue'],meta1_z['net_assets'],s=volume, c='blue', marker='o', alpha=0.8)
ax.plot_surface(x_surf,y_surf,fittedY.values.reshape(x_surf.shape), color='black', alpha=.8)
ax.view_init(elev=40., azim=160)
ax.set_xlabel('Total Expenses')
ax.set_ylabel('Total Revenue')
ax.set_zlabel('Net Assets')
plt.show()


# In[ ]:



