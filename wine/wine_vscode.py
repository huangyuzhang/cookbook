#%% [markdown]
# # 0. Basic Setup and Exploration

#%%
# Basic packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

# import sklearn packages
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
#from sklearn.neural_network import MLPRegressor
from math import sqrt

# import visualization libraries
from IPython.display import Image  
import pydotplus
from sklearn.externals.six import StringIO

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# make the output stable across runs
np.random.seed(42)


#%%
# read the data
wine_raw = pd.read_csv('wine.csv')



#%%
# glimpse the data
print("Shape of Wine data:\nrows:", wine_raw.shape[0], '\ncolumns:', wine_raw.shape[1])


#%%
wine_raw.head()


#%%
wine_raw.describe()


#%%
# check missing data (There is no missing data in the entire dataset.)
total = wine_raw.isnull().sum().sort_values(ascending = False)
percent = (wine_raw.isnull().sum()/wine_raw.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


#%%
# check data unbalance (The data has not a large unbalance with respect of the target value.)
# explore the target variable: quality
qualitydata = wine_raw.quality.value_counts().sort_index()
qualitydata_df = pd.DataFrame({'Quality': qualitydata.index,'Count': qualitydata.values})
qualitydata_df


#%%
# visualize target variable
plt.figure(figsize=(10,6))
sns.barplot(x = 'Quality', y ="Count", data = qualitydata_df,palette="Blues_d")
plt.title('Quality level Distribution',fontsize=20)
plt.show()

#%% [markdown]
# # 1. Decision Tree Classification
#%% [markdown]
# ## 1.1 Multi-class Classification

#%%
# Selecting the input and output features for tasks
features = ['fixed_acidity',
            'volatile_acidity',
            'citric_acid',
                'residual_sugar',
            'chlorides',
                'free_sulfur_dioxide',
            'total_sulfur_dioxide',
            'density',
                'ph',
            'sulphates',
            'alcohol']
target = ['quality']

X = wine_raw[features]
y = wine_raw[target]

# Visualize the combined table (which should looks the same as the original dataset)
# pd.concat([X, y], axis=1, sort=False).head()


#%%
# Split dataset into training set & test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)


#%%
from sklearn.tree import DecisionTreeClassifier

# Fit on train set
# clf = DecisionTreeClassifier(max_features=5, max_depth=9, random_state=42) # Grid Search accuracy: 0.625
clf = DecisionTreeClassifier(max_features = 5, random_state = 42, max_depth = 5) # accuracy

clf.fit(X_train, y_train)


#%%
# Test the accuracy
prediction = clf.predict(X_test)
print("Decision Tree Accuracy:",accuracy_score(y_test, prediction))


#%%
# cross validation Score
cv_clf = clf
cv_X_test = X_test
cv_y_test = y_test
cv_folds = 5
cv_scoring = None # default accuracy
cv_result = cross_val_score(cv_clf,cv_X_test,cv_y_test,cv=cv_folds,scoring=cv_scoring)
print(cv_result)
print('Mean: %.5f, Std: %.5f' % (np.mean(cv_result),np.std(cv_result)))


#%%
# Decision Tree Visualization - Multi-class
feature_names = np.array(features)
target_names = ['3','4','5','6','7','8']

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=feature_names,
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("wine_dt_1.png")
Image(graph.create_png()) 


#%%
# Feature Importance Evaluation
# ref: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_
# The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. 
# It is also known as the Gini importance.
tmp = pd.DataFrame({'Feature': features, 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (12,8))
plt.title('Features importance (Multi-class Decision Tree)',fontsize=20)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 


#%%
# print("Decision Tree Accuracy:",accuracy_score(y_test, clf.predict(X_test)))

conf_mat = confusion_matrix(y_test, clf.predict(X_test))
# plot_confusion_matrix(conf_mat, classes=class_names, title='Confusion matrix')
print('Confusion matrix:\n', conf_mat)

labels = [3,4,5,6,7,8]
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predict')
plt.title('Multi-class Decision Tree Confusion Matrix',fontsize=16)
plt.ylabel('Actual')
plt.show()


#%%
# Grid Search for Decision Tree Multi-classification
classifier = DecisionTreeClassifier(random_state = 42)
parameters = {
              'max_depth': range(3,10),
              'max_features': range(3,10),
              'max_leaf_nodes':[5,10,20,100,200],
             }
scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)

grid = GridSearchCV(classifier, parameters, scoring_fnc, cv=kfold)
grid = grid.fit(X_train, y_train)

clf = grid.best_estimator_

print('best score: %f'%grid.best_score_)
print('best parameters:')
for key in parameters.keys():
    print('%s: %d'%(key, clf.get_params()[key]))

print('test score: %f'%clf.score(X_test, y_test))

pd.DataFrame(grid.cv_results_).T

#%% [markdown]
# ## 1.2 Binary Classification

#%%
# Convert to a Binary Classification Task
# From the confusion matrix above, we can see a clear boundary between level 5 & 6
# Create a new column called Quality Label. This column will contain the values of 0 & 1
# where 1 = good, 0 = bad
wine2 = wine_raw
wine2['quality_label'] = (wine2['quality'] > 5.5)*1
wine2.head()


#%%
# explore the binary target variable: quality_label
ql_count = wine2.quality_label.value_counts().sort_index()
print(ql_count)
plt.figure(figsize=(10,6))
plt.title('Quality Label (Binary) Distribution',fontsize=16)
ql_hist = ql_count.plot(kind='bar');
ql_hist.set_xticklabels(ql_hist.get_xticklabels(),rotation=0)
plt.show() 


#%%
# Selecting the input and output features for classification tasks
features2 = ['fixed_acidity',
             'volatile_acidity',
             'citric_acid',
             'residual_sugar',
             'chlorides',
            'free_sulfur_dioxide',
             'total_sulfur_dioxide',
             'density',
             'ph',
             'sulphates',
             'alcohol']
target2 = ['quality_label']


#%%
X2 = wine2[features2]
y2 = wine2[target2]


#%%
# Split dataset
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=200)


#%%
# Fit on train set
# wine_clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, random_state=200)
clf2 = DecisionTreeClassifier(max_depth = 6, max_features = 6, random_state = 42)
clf2.fit(X2_train, y2_train)


#%%
# Measure plain accuracy of the classifier
prediction2 = clf2.predict(X2_test)
accuracy_score(y_true=y2_test, y_pred=prediction2)


#%%
# Cross Validation - Decision Tree Binary
cv_clf = clf2
cv_X_test = X2_test
cv_y_test = y2_test
cv_folds = 5
cv_scoring = None 
cv_scoring1 = 'roc_auc'
cv_result = cross_val_score(cv_clf,cv_X_test,cv_y_test,cv=cv_folds,scoring=cv_scoring)
cv_result1 = cross_val_score(cv_clf,cv_X_test,cv_y_test,cv=cv_folds,scoring=cv_scoring1)
print(cv_result)
print(cv_result1)
print('Plain: Mean %.5f, Std %.5f' % (np.mean(cv_result),np.std(cv_result)))
print('  AUC: Mean %.5f, Std %.5f' % (np.mean(cv_result1),np.std(cv_result1)))


#%%
# Decision Tree Visualisation - Binary
feature_names2 = np.array(features)
target_names2 = ['Bad','Good']

dot_data = tree.export_graphviz(clf2, out_file=None,
                         feature_names=feature_names2,
                         class_names=target_names2,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("wine_dt_2.png")
Image(graph.create_png()) 


#%%
tmp = pd.DataFrame({'Features': features2, 'Feature importance': clf2.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (12,8))
plt.title('Features importance (Binary Decision Tree)',fontsize=20)
s = sns.barplot(x='Features',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 


#%%
# Confusion Matrix - Binary Decision Tree
conf_mat = confusion_matrix(y2_test, clf2.predict(X_test))
print('Confusion matrix:\n', conf_mat)

labels = ['Bad', 'Good']
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.title('Binary Decision Tree Confusion Matrix',fontsize=16)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()


#%%
# Plot the ROC Curve
probs = clf2.predict_proba(X2_test) # Predict class probabilities of the input samples 
preds = probs[:,1]
y2_score = clf2.fit(X2_train, y2_train).predict_proba(X2_test)
 
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y2_test, preds) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

print(roc_auc)

plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, lw=2, label='AUC = %0.3f' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Binary Decision Tree ROC Curve',fontsize=20)
plt.legend(loc="lower right")
plt.show()


#%%
# Plot the precision recall curve
precisions, recalls, thresholds = precision_recall_curve(y2_test, preds)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([0, 1.1])
    
plt.figure(figsize=(10,6))
plt.title('Decision Tree Precision Recall Curve',fontsize=20)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([0, 1])
plt.show()


#%%
# Grid Search for Decision Tree Binary
classifier = DecisionTreeClassifier(random_state = 42)
parameters = {
              'max_depth': range(3,10),
              'max_features': range(3,10),
              'max_leaf_nodes':[5,10,20,100,200],
             }
scoring_fnc = make_scorer(accuracy_score)
kfold = KFold(n_splits=10)

grid = GridSearchCV(classifier, parameters, scoring_fnc, cv=kfold)
grid = grid.fit(X2_train, y2_train)

clf = grid.best_estimator_

print('best score: %f'%grid.best_score_)
print('best parameters:')
for key in parameters.keys():
    print('%s: %d'%(key, clf.get_params()[key]))

print('test score: %f'%clf.score(X2_test, y2_test))

pd.DataFrame(grid.cv_results_).T

#%% [markdown]
# # 2. Random Forest
#%% [markdown]
# ## 2.1 Multi-class

#%%
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)


#%%
print("Multi-class Random Forest Accuracy:", accuracy_score(y_test, rf_clf.predict(X_test)))


#%%
# cross validation Score - Multi-class Random Forest
cv_clf = rf_clf
cv_X_test = X_test
cv_y_test = y_test
cv_folds = 5
cv_scoring = None # default accuracy
cv_result = cross_val_score(cv_clf,cv_X_test,cv_y_test,cv=cv_folds,scoring=cv_scoring)
print('cross validation - Random Forest Multi-class')
print(cv_result)
print('Mean: %.5f, Std: %.5f' % (np.mean(cv_result),np.std(cv_result)))


#%%
tmp = pd.DataFrame({'Feature': features, 'Feature importance': rf_clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (12,8))
plt.title('Features importance (Multi-class Random Forest)',fontsize=20)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()


#%%
# plot the confusion matrix
conf_mat = confusion_matrix(y_test, rf_clf.predict(X_test))
print('Confusion matrix:\n', conf_mat)

labels = [3,4,5,6,7,8]
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.title('Multi-class Random Forest Confusion Matrix',fontsize=16)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()

#%% [markdown]
# ## 2.2 Binary

#%%
rf_clf2 = RandomForestClassifier()
rf_clf2.fit(X2_train, y2_train)


#%%
print("Binary Random Forest Accuracy:", accuracy_score(y2_test, rf_clf2.predict(X2_test)))


#%%
# Cross Validation - Random Forest Binary
cv_clf = rf_clf2
cv_X_test = X2_test
cv_y_test = y2_test
cv_folds = 5
cv_scoring = None 
cv_scoring1 = 'roc_auc'
cv_result = cross_val_score(cv_clf,cv_X_test,cv_y_test,cv=cv_folds,scoring=cv_scoring)
cv_result1 = cross_val_score(cv_clf,cv_X_test,cv_y_test,cv=cv_folds,scoring=cv_scoring1)
print(cv_result)
print(cv_result1)
print('Plain: Mean %.5f, Std %.5f' % (np.mean(cv_result),np.std(cv_result)))
print('  AUC: Mean %.5f, Std %.5f' % (np.mean(cv_result1),np.std(cv_result1)))


#%%
tmp = pd.DataFrame({'Features': features2, 'Feature importance': rf_clf2.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (12,8))
plt.title('Features importance (Binary Random Forest)',fontsize=20)
s = sns.barplot(x='Features',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()


#%%
conf_mat = confusion_matrix(y2_test, rf_clf2.predict(X2_test))
print('Confusion matrix:\n', conf_mat)

labels = ['Bad', 'Good']
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.title('Binary Random Forest Confusion Matrix',fontsize=16)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()


#%%
from sklearn.model_selection import cross_val_predict, cross_val_score
cv = 5
print('Score(AUC): %.5f +/- %.4f' % (np.mean(cross_val_score(rf_clf2,X2_test,y2_test,cv=cv,scoring='roc_auc')),np.std(cross_val_score(rf_clf2,X2_test,y2_test,cv=cv,scoring='roc_auc'))))


#%%
# Plot the ROC Curve
rf_probs = rf_clf2.predict_proba(X2_test) # Predict class probabilities of the input samples 
rf_preds = rf_probs[:,1]
 
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y2_test, rf_preds) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, lw=2, label='AUC = %0.3f' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=20)
plt.legend(loc="lower right")
plt.show()


#%%
# Plot the precision recall curve
precisions, recalls, thresholds = precision_recall_curve(y2_test, rf_preds)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.ylim([0, 1.1])
    
plt.figure(figsize=(10,6))
plt.title('Random Forest Precision Recall Curve',fontsize=20)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([0, 1])
plt.show()

#%% [markdown]
# # 3. Conclusion
#%% [markdown]
# - The dataset is relatively small, 1599 observations. (A larger dataset is found with 6497 observations, similar result);
# - The dataset is special so the fields might not be widely available in industry wise;
# - For the future use, we might have limited fields data available so the model could not work as this dataset, for example 3 most important variables: alcohol, sulphates, volatile acidity;
# - Use grid search to optimise the hyper-parameters of Random Forest;

#%%



