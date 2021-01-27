#The goal of this project is to create a model that predicts the probability an NFL player will reach the Pro Bowl
#based on the player's performance in the NFL combine
#Data Source:http://nflcombineresults.com/nflcombinedata_expanded.php?year=2001&pos=&college=

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

#===============OBTAIN DATA===================
#read data into a dataframe
combine = pd.read_csv('data/nfl_combine_data.csv')
pro_bowl = pd.read_csv('data/probowlers.csv')

#clean the data
combine.rename(columns=lambda x: x.strip(), inplace=True) #remove whitespace
combine.dropna(how='all', axis='columns',inplace=True) #remove empty columns

pro_bowl['Player'] = pro_bowl['Player'].str.replace('[%,+]','')
pro_bowl.rename(columns=lambda x: x.strip(), inplace=True) #remove whitespace
pro_bowl.dropna(how='all', axis='columns',inplace=True) #remove empty columns

#get list of all probowlers
pro_bowl_players = pro_bowl['Player'].drop_duplicates()

#join combine data with pro bowl data
full = combine.merge(pro_bowl_players,how='left',left_on='Name',right_on='Player')

#mark all probowlers
full['Is Pro Bowl'] = full['Player'].apply(lambda x: 0 if x!=x else 1)


#=================EXPLORE DATA================
#QUESTION: Can one predict pro bowl liklihood from NFL combine performance?
#Start with corner back, most influenced by raw athleticism

#let's look at CB's only
cornerbacks = full[full['POS']=='CB']

#visualize the data

#height
plt.style.use('seaborn-deep')
cb_pro = cornerbacks[cornerbacks['Is Pro Bowl'] == 1]['Height (in)'] #np.random.normal(1, 2, 5000)
cb_non_pro = cornerbacks[cornerbacks['Is Pro Bowl'] == 0]['Height (in)']  #np.random.normal(-1, 3, 2000)
bins = np.linspace(60, 80, 20)


#plot 40 yard dash times for CBs
plt.figure(figsize=(14, 8))
sns.catplot(x="Is Pro Bowl", y="40 Yard", kind="violin", data=cornerbacks, height=8.5, aspect=.9)

#================Split Data into Test and Train sets======================


#================Select Features===================



#================fit logistic regression===================
#split data into training set and test set



#================evaluat on test data===================
#split data into training set and test set


#==========iterate model as needed======================


#==========deploy======================

