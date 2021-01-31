#The goal of this project is to create a model that predicts the probability an NFL player will reach the Pro Bowl
#based on the player's performance in the NFL combine
#Data Source:http://nflcombineresults.com/nflcombinedata_expanded.php?year=2001&pos=&college=

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

#===============OBTAIN DATA===================
#read data into a dataframe
combine = pd.read_csv('data/nfl_combine_data.csv')
pro_bowl = pd.read_csv('data/probowlers.csv')

#clean the data
combine.rename(columns=lambda x: x.strip(), inplace=True) #remove whitespace
combine.columns = combine.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '') #remove empty columns
combine.dropna(how='all', axis='columns',inplace=True) #remove extraneous empty columns
combine.columns = combine.columns.str.replace(u'\xa0', ' ')#remove non-breaking space
combine.drop(columns=['wonderlic'],inplace=True) #remove wonderlic as there is litle data
combine = combine[combine['year'] != 2020] #drop 2020 combine data since those players could not have made 2020 pro bowl

pro_bowl['Player'] = pro_bowl['Player'].str.replace('[%,+]','')
pro_bowl.rename(columns=lambda x: x.strip(), inplace=True) #remove whitespace
pro_bowl.dropna(how='all', axis='columns',inplace=True) #remove empty columns

#get list of all probowlers
pro_bowl_players = pro_bowl['Player'].drop_duplicates()

#join combine data with pro bowl data
full = combine.merge(pro_bowl_players,how='left',left_on='name',right_on='Player')

#mark all probowlers
full['is_pro_bowl'] = full['Player'].apply(lambda x: 0 if x!=x else 1)

#=================EXPLORE================
#QUESTION: Can one predict pro bowl liklihood from NFL combine performance?
#Start with corner back, most influenced by raw athleticism

#let's look at CB's only
cornerbacks = full[full['pos']=='CB']


#look at count of probowlers versus non-PBs
sns.countplot(x='is_pro_bowl',data=cornerbacks) #,palette='hls')
plt.show()
plt.savefig('count_plot.png')


#find percentages of probowl cornerbacks
count_no_pb = len(cornerbacks[cornerbacks['is_pro_bowl']==0])
count_pb = len(cornerbacks[cornerbacks['is_pro_bowl']==1])
pct_of_no_pb = count_no_pb/(count_no_pb + count_pb)
print("percentage of not pro bowlers: ", pct_of_no_pb*100)
pct_of_pb = count_pb/(count_no_pb+count_pb)
print("percentage of pro bowlers: ", pct_of_pb*100)


#lets see the means across columns for the two groups
print(cornerbacks.groupby('is_pro_bowl').mean())


#=================VISUALIZE================
#columns to plot

plots = ['height in', 'weight lbs', 'hand_size in',
         'arm_length in', '40 yard', 'bench_press','vert_leap in',
         'broad_jump in', 'shuttle', '3cone', '60yd_shuttle']
#plot all the columns
for plot in plots:
       sns.catplot(x="is_pro_bowl", y = plot, kind="violin", data=cornerbacks, height=8.5, aspect=.9)
       plt.savefig(plot + '.png')


#============Hypothesis Testing======================
from scipy.stats import ttest_ind
for plot in plots:
    t1 = cornerbacks[cornerbacks['is_pro_bowl'] == 1][plot]
    t2 = cornerbacks[cornerbacks['is_pro_bowl'] == 0][plot]
    t1_mean = np.mean(t1)
    t2_mean = np.mean(t2)
    t1_std = np.std(t1)
    t2_std = np.std(t2)
    ttest,pval = ttest_ind(t1, t2)


    if pval <0.05:
        print('====== ', plot, " ======")
        print("p-value", pval)
        print("we reject null hypothesis [" + plot + "] IS sig different!")
        print("Pro Bowl mean value:",t1_mean," stdev: ",t1_std)
        print("Non- PB mean value:", t2_mean," stdev: ",t2_std)
        print("==== END ", plot, " ====")
        print(" ")
    #else:
        #print(" ") #we accept null hypothesis " + plot + " IS NOT sig diff")


#=========================================
#==============Logistic Regression========
#=========================================

#drop categorical columns
cornerbacks = cornerbacks[['height in', 'weight lbs', 'hand_size in',
         'arm_length in', '40 yard', 'bench_press','vert_leap in',
         'broad_jump in', 'shuttle', '3cone', '60yd_shuttle','is_pro_bowl']]

#====impute NaN values with mean for use in model===============
cornerbacks.dropna(axis=0, how='any', thresh=5, inplace=True) # drop some rows with little data
cornerbacks = cornerbacks.fillna(cornerbacks.mean())


#define X and y; features and results
X = cornerbacks.loc[:, cornerbacks.columns != 'is_pro_bowl']
y = cornerbacks.loc[:, cornerbacks.columns == 'is_pro_bowl']


#==========OVER SAMPLING USING SMOTE=================

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

os_data_X, os_data_y = os.fit_sample(X_train, y_train) #oversample training data
os_data_X = pd.DataFrame(data=os_data_X,columns=columns)
os_data_y= pd.DataFrame(data=os_data_y,columns=['is_pro_bowl'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of non-Pro Bowlers in oversampled data",len(os_data_y[os_data_y['is_pro_bowl']==0]))
print("Number of Pro Bowlers",len(os_data_y[os_data_y['is_pro_bowl']==1]))
print("Proportion of non-Pro Bowler data in oversampled data is ",len(os_data_y[os_data_y['is_pro_bowl']==0])/len(os_data_X))
print("Proportion of Pro Bowler data in oversampled data is ",len(os_data_y[os_data_y['is_pro_bowl']==1])/len(os_data_X))


#use all columns based on RFE results
cols=['height in', 'weight lbs', 'hand_size in', 'arm_length in', '40 yard',
       'bench_press', 'vert_leap in', 'broad_jump in', 'shuttle', '3cone',
       '60yd_shuttle']
X=os_data_X[cols]
y=os_data_y['is_pro_bowl']

#===========Check Model Characteristics==========================

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


#==========LOGISTIC REGRESSION MODEL FIT=============
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(C=200, class_weight=None, dual=False,
                            fit_intercept=True, intercept_scaling=1,
                            max_iter=1000, multi_class='ovr',n_jobs=1,
                            penalty='l2',random_state=None, solver='liblinear',
                            tol=0.0001,verbose=0,warm_start=False)
logreg.fit(X, y)

#predicting test set results and calculating accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test, y_test)))

#==========PRECISION, RECALL, F-MEASURE, SUPPORT=====
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#==========CONFUSION MATRIX==========================
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

#plot confusion matrix
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()
plt.savefig('confusion_matrix.png')


#==========ROC CURVE=================================
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
