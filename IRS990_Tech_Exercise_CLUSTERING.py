
# coding: utf-8

# # Data Insights Take-Home: Clustering on Expenses and Revenues
# **Marianne C. Halloran
# October 12, 2017**
# <br>
# <br>
# <p>*** The principal value of detailing the financial information obtained in Form 990 is to bring insight and arrive at data-backed conclusions about the NPOs, and their ability to garner financial support to continue operations. ***
# Here, The idea is that, by understanding how a NPO obtains revenue and spends its funds, we will be better poised to understand its efficacy. It also answers the questions of the financial strength of the NPO (its ability to attract resources, level of reserves, financial accountability, etc).
# 
# #### Data selection rationale and visualizations
# First, I generate visualizations of the sources of income for 501(c)(3) NPOs, based on fundraising, campaign, membership, government grants, gifts and service revenues. This can provide insights into the income nature of the NPOs. Some NPOs can receive most of their funds from chargings fees, or through government grants. *To some individuals, this can often play an important factor in their donation decision*.
# 
# Similarly, I generate visualizations of the expenses, based on functional, service, management, and fundraising expenses. Individuals interested in NPOs can be interested in how the NPOs are spending most of its resources on program matters and not on management or fundraising, for example. 
# 
# Net assets provide some indication of the level of resources the filer has to help support its activities in the future. 
# 
# Moreover, compensation of its employees versus its income and expenditure can bring important information about the NPOs and their financial health and resource allocation.
# 
# Here, I perform a basic clustering for three features: ***total compensation***, **total income plus assets**, and ***total expenses***. The idea behind this selection would  be to identify similar NPOs and the relationship between the three features: is there an inherent separation in the data?
# 
# #### Pre-processing and Clustering
# Data was normalized using z-scores.
# 	
# I chose ***z-score normalization*** although all data is given in dollars, these values necessarily comparable. Standardizing them using z-scores is a best-practice to give it equal weights by minimizing the error function using the Newton algorithm, i.e. a gradient-based optimization algorithm. Normalizing the data improves convergence of such algorithms.
# 
# **Suggestions:**
# 
# - Financial information is more meaningful if viewed over a period of several years, seeing how organizations can change over time. A single year's Form 990 provides only a snapshot in time. 

# In[1]:

#=================================================================#
# LIBRARIES                                                       #
#=================================================================#
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from pylab import rcParams
from mpl_toolkits.mplot3d import Axes3D

get_ipython().magic(u'matplotlib inline')
rcParams['figure.figsize'] = 8,8


# In[2]:

#=================================================================#
# DATA IMPPORT                                                    #
#=================================================================#
meta = pd.read_csv('input/NPO_meta_38k.csv')
meta.columns = ['EIN','contract_term','tax_status','org_name','city','state','tax_year',
                'activity','year_formed','volunteer_ct','employee_ct','rev_campaigns',
                'rev_membership', 'rev_fundraising','rev_govgrants','rev_other','rev_progserv',
                'rev_netfundraising','total_revenue','total_revenuePY','exp_grants','exp_progserv',
                'exp_management','exp_fundraising','total_expenses','total_compensations',
                'comp_more100k', 'net_assets','pol_act','lob_act','foreign_office',
                'foreign_fundraising','foreign_assist']
del meta['EIN'],  meta['contract_term']# meta['activity'],meta['year_formed'],
print(u"\u0011",'Cleaned data, removed NaN')


# I'm removing any organization that is not a 501(c)(3) and any orgs with NaN in a row
meta = meta.dropna(axis=0,how='any')
meta_501c3 = meta.loc[meta['tax_status'] == 0]
del meta; meta = meta_501c3
meta


# In[3]:

#=================================================================#
# DESCRIPTIVE STATISTICS                                          #
#=================================================================#
print(u"\u0011",'Descriptive statistics, summarizing central tendency, dispersion')
print('  and shape of dataset\'s distribution')
meta.describe()


# In[4]:

#=================================================================#
# PROCESS DATA: Categorical conversions, OHE, features            #
#=================================================================#
# Cities and States will get categorical codes
meta['city'] = meta['city'].str.upper() # all upper case
cities = sorted(meta['city'].unique())  # sort by unique names
meta['city_int'] = meta['city'].map(lambda x: cities.index(x))


# In[5]:

#=================================================================#
# VISUALIZATION                                                   #
#=================================================================#
# Visualizations of the sources of income for 501(c)(3) NPOs, 
# based on fundraising, campaign, membership, government grants, 
# gifts, assets and service revenues. This can provide insights into the 
# income nature of the NPOs. 
print(u"\u0011","It is interesting that in average, most NPOs have almost zero profitability (Income minus Expense)")
fig = plt.figure()
rev_df = meta[['total_revenue', 'total_expenses', 'net_assets']].copy()
ax = sns.barplot(data=rev_df)
ax.set(xlabel='Revenue', ylabel='Dollars (US$)')
ax.set_xticklabels(['Total Revenue','Total Expenses','Net Assets'], rotation=30)
ax.set_title('Total Revenue, Expenses and Net Assets', fontsize=16)
plt.show()

del rev_df
# Revenue Plot, normalized by total revenue
print(u"\u0011","Note that most NPOs' income comes from Program Services (23%)", 
      "followed closely by income from Other sources (Gifts, Donations,etc) at 21%. ",
      "Government Grants only account for 8.8% of the total revenue")
fig = plt.figure()
rev_df = meta[['rev_campaigns','rev_membership','rev_fundraising','rev_netfundraising',
               'rev_govgrants','rev_progserv','rev_other']].copy()

rev_df=(rev_df.div(meta['total_revenue'], axis=0)).fillna(0)
ax = sns.barplot(data=rev_df)
for p in ax.patches:
    ax.annotate("%.2f" % (p.get_height()*100),
                (p.get_x() + p.get_width() / 2., .02), 
                fontsize=16,ha='center', va='bottom')
ax.set_xlabel('Revenue', fontsize=14)
ax.set_xticklabels(['Campaigns','Membership','Fundraising','Other Fundraising',
                    'Government Grants', 'Program Services','Other'], rotation=30, fontsize=14)
ax.set_title('Percentage of Total Revenue for each source', fontsize=16)
ax.set_ylabel('Dollars (US$)',fontsize=14)
plt.show()


# In[6]:

# Visualizations of the expenses, based on functional, service, 
# management, and fundraising expenses. 
# Normalized by total expenses
print(u"\u0011","Here, we see that, while Grants and Fundraising constitute only 8% of the expenses,",
      "Management/Functional and Compensations costs account, in average, for 22% of expenses.")
fig = plt.figure();
exp_df = meta[['exp_grants','exp_progserv','exp_management',
               'exp_fundraising', 'total_compensations']].copy()
exp_df=(exp_df.div(meta['total_expenses'], axis=0)).fillna(0)

ax = sns.barplot(data=exp_df)
for p in ax.patches:
    ax.annotate("%.2f" % (p.get_height()*100),
                (p.get_x() + p.get_width() / 2., .02), 
                fontsize=16,ha='center', va='bottom')
ax.set_xlabel('Expenses', fontsize=14)
ax.set_xticklabels(['Grants','Program Services','Management/Functional','Fundraising', 'Compensations'], rotation=20, fontsize=14)
ax.set_title('Percentage of Total Expenses for each source', fontsize=16)
ax.set_ylabel('Dollars (US$)',fontsize=14)
plt.show()


# In[7]:

#=================================================================#
# VISUALIZATION                                                   #
#=================================================================#
## Create array for K-means
# Standarize (z-score) array (zi = xi-xmean/std)
meta_ = (meta[['net_assets', 'total_revenue']].copy()).apply(stats.zscore)
meta_zscored = meta_[(np.abs(stats.zscore(meta_)) < 3).all(axis=1)]

## Visualizations
plt.scatter(meta_['net_assets'], meta_['total_revenue'], s=80);
plt.title('Scatter plot of Employee Count vs Total Revenue', fontsize=16)
plt.xlabel('Net Assets (US$)', fontsize=14); plt.ylabel('Total Revenue (US$)', fontsize=14); 
plt.show()

plt.scatter(meta_zscored['net_assets'], meta_zscored['total_revenue'], s=80);
plt.title('[Z-score Normalized] Scatter plot of Employee Count vs Total Revenue', fontsize=16)
plt.xlabel('Net Assets (US$)', fontsize=14); plt.ylabel('Total Revenue (US$)', fontsize=14); 
plt.show()


# In[8]:

#=================================================================#
# 3D VISUALIZATION                                                #
#=================================================================#
# Add columns City_int and State_int to processed data
meta_ = (meta[['total_compensations', 'total_revenue', 'total_expenses']].copy()).apply(stats.zscore)
meta_zscored = meta_[(np.abs(stats.zscore(meta_)) < 2).all(axis=1)]

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(meta_zscored['total_compensations'],meta_zscored['total_revenue'],meta_zscored['total_expenses'], s=80)
ax.set_xlabel('Total Compensation (z-scored)', fontsize=14)
ax.set_ylabel('Total Revenue (zscore)', fontsize=14)
ax.set_zlabel('Total Expenses (zscore)', fontsize=14)
ax.set_title('Revenue, Compensation and Expenses', fontsize=16)
ax.view_init(elev=5., azim=60)
plt.show()


# In[9]:

#=================================================================#
# K-Means Clustering                                              #
#=================================================================#
# Elbow method to determine K
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(meta_zscored)
    kmeanModel.fit(meta_zscored)
    distortions.append(sum(np.min(cdist(meta_zscored, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / meta_zscored.shape[0])
plt.plot(K, distortions, 'bx-')
plt.xlabel('k', fontsize=14)
plt.ylabel('Distortion', fontsize=14)
plt.title('Elbow Method for Optimal k', fontsize=20)
plt.show()
 
#=================================================================#
print(u"\u0011","Ideally, we would use a more stringent criterion determination method, such as",
      "the Akaike information criterion (AIC) or Bayesian information criterion (BIC)\n.")
# KMeans 
kmeans = KMeans(n_clusters=4, random_state=0).fit(meta_zscored)
labels = kmeans.labels_
## Add labels to original data
meta_zscored = meta_zscored.assign(Clusters = labels)
columns = (meta_zscored.columns.get_values()).tolist()
print(u"\u0011","For k=4:",meta_zscored[columns].groupby(['Clusters']).mean())


# In[10]:

#=================================================================#
# VISUALIZATION K-Means Clustering                                #
#=================================================================#
# Generate cluster groupings
cluster1=meta_zscored.loc[meta_zscored['Clusters'] == 0]
cluster2=meta_zscored.loc[meta_zscored['Clusters'] == 1]
cluster3=meta_zscored.loc[meta_zscored['Clusters'] == 2]
cluster4=meta_zscored.loc[meta_zscored['Clusters'] == 3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cluster1['total_compensations'],cluster1['total_revenue'],cluster1['total_expenses'], 
           c='blue', s=40, cmap="RdBu")
ax.scatter(cluster2['total_compensations'],cluster2['total_revenue'],cluster2['total_expenses'], 
           c='red', s=40, cmap="RdBu")
ax.scatter(cluster3['total_compensations'],cluster3['total_revenue'],cluster3['total_expenses'], 
           c='green', s=40, cmap="RdBu")
ax.scatter(cluster4['total_compensations'],cluster4['total_revenue'],cluster4['total_expenses'], 
           c='yellow', s=40, cmap="RdBu")
ax.set_xlabel('Total Compensation (z-scored)', fontsize=14)
ax.set_ylabel('Total Revenue (zscore)', fontsize=14)
ax.set_zlabel('Total Expenses (zscore)', fontsize=14)
ax.set_title('KMeans Clustering for Revenue, Compensation and Expenses', fontsize=16)
ax.view_init(elev=5., azim=60)
plt.show()


# In[11]:

# Retrieve original data and clean NaNs due to zscore outlier removal
meta = meta.assign(Clusters = meta_zscored['Clusters'].loc[meta_zscored.index.get_values()])
meta = meta.dropna(axis=0,how='any')


# In[12]:

#=================================================================#
# CLUSTER 1 ANALYSIS                                              #
#=================================================================#
print(u"\u0011","Cluster 1 contains ",len(cluster1), "companies", "with an average of 9.84 Employees and Average Compensation (Total) of US$19,137.95, with a mean of 0.049 employees receiving salaries above U$100k. ")
sns.jointplot(x="comp_more100k", y="employee_ct", data=meta.loc[meta['Clusters'] == 0]); 
plt.title('Cluster 1', fontsize=20); plt.show()
(meta.loc[meta['Clusters'] == 0]).describe()


# In[13]:

#=================================================================#
# CLUSTER 2 ANALYSIS                                              #
#=================================================================#
print(u"\u0011","Cluster 2 contains",len(cluster2), "companies", "with an average of 246 Employees and Average Compensation (Total) of US$1,023,683.00, with a mean of 7.96 employees receiving salaries above U$100k. ")
sns.jointplot(x="comp_more100k", y="employee_ct", data=meta.loc[meta['Clusters'] == 1]); 
plt.title('Cluster 2', fontsize=20); plt.show()
(meta.loc[meta['Clusters'] == 1]).describe()


# In[14]:

#=================================================================#
# CLUSTER 3 ANALYSIS                                              #
#=================================================================#
print(u"\u0011","Cluster 3 contains ",len(cluster3), "companies", "with an average of 140 Employees and Average Compensation (Total) of US$335,620.96, with a mean of 1.86 employees receiving salaries above U$100k. ")
sns.jointplot(x="comp_more100k", y="employee_ct", data=meta.loc[meta['Clusters'] == 2]); 
plt.title('Cluster 3', fontsize=20); plt.show()
(meta.loc[meta['Clusters'] == 2]).describe()


# In[15]:

#=================================================================#
# CLUSTER 4 ANALYSIS                                              #
#=================================================================#
print(u"\u0011","Similar to Cluster 3, Cluster 4 contains ",len(cluster4), "companies", "with an average of 647 Employees and Average Compensation (Total) of US$927,991.6e+05, with a mean of 14.4 employees receiving salaries above U$100k. ")
sns.jointplot(x="comp_more100k", y="employee_ct", data=meta.loc[meta['Clusters'] == 3]); 
plt.title('Cluster 4', fontsize=20); plt.show()
(meta.loc[meta['Clusters'] == 3]).describe()


# ## Results from Clustering
# 
# My analysis identified 4 clusters of companies in database. As shown above, Clusters 1 and 3 contain companies that are relatively small (9.84 and 140 employees in average, respectively), but counted with only, in average 0.049 and 1.86 of its employees receiving salaries over U\$100,000, respectively. 
# 
# On the other hand, Clusters 2 and 4 show NPOs with a comparatively larger number of employees (246 and 647 in average, respectively), however, its average number of employees receiving remuneration of U$100,000 and higher exceeds Clusters 1 and 3 117-fold (0.095 versus 11.18 average for Clusters 1,3 and Clusters 2,4, respectively). The relationship between the Clusters and their Compensations and Revenues can be seen in Result Figures 1 and 2, below.
# 
# In Result Figure 2 and 3, I note that although NPOs in Cluster 4 are able to attract a larger amount of income, its number of volunteers varies greatly within the dataset, and is not much different from Clusters 2 and 3.
# 
# In addition, the three graphs below, show that revenues and expenses are also tied together for these clusters, and that the NPO clusters with highers revenues/expenses also count with the largest number of volunteers. 
# Taken together, this data can help individuals and organizations best analyze the financial resources and its uses by the NPOs analyzed.

# In[16]:

# Visualizations of the expenses, based on functional, service, 
# management, and fundraising expenses. 
# Normalized by total expenses
fig = plt.figure();
exp_df = meta[['total_compensations','Clusters']].copy()
exp_DF = exp_df.groupby(['Clusters'])
ax = sns.barplot(data=exp_df, x="Clusters", y="total_compensations", order=[0,1,2,3])
for p in ax.patches:
    ax.annotate("~U$%.0f" % (p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()*1.08), 
                fontsize=16,ha='center', va='bottom')
ax.set_xlabel('Clusters', fontsize=14)
ax.set_xticklabels(['Cluster 1','Cluster 2', 'Cluster 3 ','Cluster 4'], rotation=20, fontsize=14)
ax.set_title('[Result Figure 1] Total Compensations for each cluster', fontsize=16)
ax.set_ylabel('Dollars (US$)',fontsize=14)
plt.show()


# In[17]:

# Visualizations of the expenses, based on functional, service, 
# management, and fundraising expenses. 
# Normalized by total expenses
fig = plt.figure();
exp_df = meta[['total_revenue','Clusters']].copy()
exp_DF = exp_df.groupby(['Clusters'])
ax = sns.barplot(data=exp_df, x="Clusters", y="total_revenue", order=[0,1,2,3])
for p in ax.patches:
    ax.annotate("~U$%.0f" % (p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()*1.08), 
                fontsize=16,ha='center', va='bottom')
ax.set_xlabel('Clusters', fontsize=14)
ax.set_xticklabels(['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4'], rotation=20, fontsize=14)
ax.set_title('[Result Figure 2] Total Revenue for each cluster', fontsize=16)
ax.set_ylabel('Dollars (US$)',fontsize=14)
plt.show()


# In[19]:

# Visualizations of the expenses, based on functional, service, 
# management, and fundraising expenses. 
# Normalized by total expenses
fig = plt.figure();
exp_df = meta[['volunteer_ct','Clusters']].copy()
exp_DF = exp_df.groupby(['Clusters'])
ax = sns.barplot(data=exp_df, x="Clusters", y="volunteer_ct", order=[0,1,2,3])
for p in ax.patches:
    ax.annotate("%.0f" % (p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height()*1.08), 
                fontsize=16,ha='center', va='bottom')
ax.set_xlabel('Clusters', fontsize=14)
ax.set_xticklabels(['Cluster 1','Cluster 2', 'Cluster 3','Cluster 4'], rotation=20, fontsize=14)
ax.set_title('[Result Figure 3] Volunteer Amounts for each cluster', fontsize=16)
ax.set_ylabel('Number of Volunteers',fontsize=14)
plt.show()


# In[ ]:



