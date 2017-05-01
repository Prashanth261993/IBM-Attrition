import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit


attritionData = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
attritionData.isnull().any()
#Observed that there are no null values in the dataset


# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)

# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

sns.distplot(attritionData.MonthlyIncome[attritionData.Gender == 'Male'], bins = np.linspace(0,20000,60))
sns.distplot(attritionData.MonthlyIncome[attritionData.Gender == 'Female'], bins = np.linspace(0,20000,60))
plt.legend(['Male','Female'])
plt.show()

# Plotting the Kernel Density Estimate of different features to understand direct correlation and finding useful patterns/features
f, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=False, sharey=False)

s = np.linspace(0, 3, 10)
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)
x = attritionData['JobSatisfaction'].values
y = attritionData['PercentSalaryHike'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])
axes[0,0].set( title = 'Percent Salary Hike vs Job Satisfaction')

cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
x = attritionData['YearsSinceLastPromotion'].values
y = attritionData['RelationshipSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])
axes[0,1].set( title = 'Years Since Last Promotion against Relationship Satisfaction')

cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
x = attritionData['YearsInCurrentRole'].values
y = attritionData['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])
axes[0,2].set( title = 'Years in role against Job Satisfaction')

cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
x = attritionData['DailyRate'].values
y = attritionData['DistanceFromHome'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,0])
axes[1,0].set( title = 'Daily Rate against DistancefromHome')

cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
x = attritionData['DailyRate'].values
y = attritionData['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])
axes[1,1].set( title = 'Daily Rate against Job satisfaction')

cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)
x = attritionData['YearsAtCompany'].values
y = attritionData['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,2])
axes[1,2].set( title = 'Years at Company against Job satisfaction')

cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
x = attritionData['YearsAtCompany'].values
y = attritionData['DailyRate'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])
axes[2,0].set( title = 'Years at company against Daily Rate')

cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)
x = attritionData['RelationshipSatisfaction'].values
y = attritionData['YearsWithCurrManager'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,1])
axes[2,1].set( title = 'Relationship Satisfaction vs years with manager')

cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)
x = attritionData['WorkLifeBalance'].values
y = attritionData['JobSatisfaction'].values
sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,2])
axes[2,2].set( title = 'WorklifeBalance against Satisfaction')

f.tight_layout()
f.show()

# Converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]



# Finding correlation between numerical features, helpful in dimensionality reduction
numerical = [u'Age', u'DailyRate', u'DistanceFromHome', u'Education', u'EnvironmentSatisfaction',
             u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',
             u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
             u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',
             u'StockOptionLevel', u'TotalWorkingYears',
             u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',
             u'YearsInCurrentRole', u'YearsSinceLastPromotion',
             u'YearsWithCurrManager']

# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

sns.set(context="paper", font="monospace")

# Load the datset of correlations between cortical brain networks

correlatonMatrix = attritionData[numerical].corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(correlatonMatrix, vmax=.8, square=True, cmap="YlGnBu")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
f.show()


# Pairplot of different features with regression fitted line
numerical = [u'Age', u'DailyRate', u'MonthlyIncome', u'YearsAtCompany', u'Attrition_numerical']
sns.set(style="ticks", color_codes=True)

g = sns.pairplot(attritionData[numerical], hue='Attrition_numerical', kind="reg")
g.set(xticklabels=[])
sns.plt.show()

attritionData = attritionData.drop(['Attrition_numerical'], axis=1)


# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numericalColumns = attritionData.columns.difference(categorical)
attritionCategorical = attritionData[categorical]
attritionCategorical = attritionCategorical.drop(['Attrition'], axis=1)
attritionCategorical = pd.get_dummies(attritionCategorical)
attritionNumerical = attritionData[numerical]
attritionFinal = pd.concat([attritionNumerical, attritionCategorical], axis=1)

# Split data into train and test sets as well as for validation and testing
trainForest, testForest, target_trainForest, target_valForest = train_test_split(attritionFinal, target, train_size= 0.75, random_state=0)

oversampler=SMOTE(random_state=0)
oversample_trainForest, oversample_targetForest = oversampler.fit_sample(trainForest, target_trainForest)


seed = 0
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 800,
    'warm_start': True,
    'max_features': 0.3,
    'max_depth': 9,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : seed,
    'verbose': 0
}
rf = RandomForestClassifier(**rf_params)
rf.fit(oversample_trainForest, oversample_targetForest)

rf_predicted = rf.predict(testForest)
metrics.confusion_matrix(target_valForest, rf_predicted, labels=None, sample_weight=None)
metrics.accuracy_score(target_valForest, rf_predicted)
scores = cross_val_score(rf, attritionFinal, target, scoring='accuracy', cv=10)
#85.85

#Scaling numerical columns
attritionNumericalScaled = pd.DataFrame(StandardScaler().fit_transform(attritionNumerical))
attritionNumericalScaled.columns = attritionNumerical.columns
attrition_factored = pd.concat([attritionNumericalScaled, attritionCategorical], axis=1)

# Split data into train and test sets as well as for validation and testing
trainLogisitc, testLogistic, target_trainLogisitc, target_valLogistic = train_test_split(attrition_factored, target, train_size= 0.75,random_state=0)

oversampler=SMOTE(random_state=0)
oversample_trainLogistic, oversample_targetLogistic = oversampler.fit_sample(trainLogisitc, target_trainLogisitc)

# Logistic Regression Model
lgModel = LogisticRegression()
lgModel = lgModel.fit(oversample_trainLogistic, oversample_targetLogistic)
log_predicted = lgModel.predict(testLogistic)
metrics.confusion_matrix(target_valLogistic, log_predicted, labels=None, sample_weight=None)
metrics.accuracy_score(target_valLogistic, log_predicted)

scores = cross_val_score(lgModel, attrition_factored, target, scoring='accuracy', cv=10)
#88

# Factoring categorical varialbes within same column instead of creating dummy columns
for column in categorical:
    x_set = set(attritionData[column])
    counter = 0
    for values in x_set:
        attritionData[column][attritionData[column] == values] = counter
        counter += 1

attritionCategoricalColumn = attritionData[categorical]
attritionCategoricalColumn = attritionCategoricalColumn.drop(['Attrition'], axis=1)
attrition_factored_column = pd.concat([attritionNumericalScaled, attritionCategoricalColumn], axis=1)
train, test, target_train, target_val = train_test_split(attrition_factored_column, target, train_size= 0.75, random_state=0)

oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_sample(train,target_train)

model = LogisticRegression()
model = model.fit(smote_train, smote_target)
predicted = model.predict(test)
metrics.confusion_matrix(target_val, predicted, labels=None, sample_weight=None)
metrics.accuracy_score(target_val, predicted)
scores = cross_val_score(model, attrition_factored_column, target, scoring='accuracy', cv=10)
#86.59


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(train, target_train).predict(test)
metrics.confusion_matrix(target_val, y_pred, labels=None, sample_weight=None)
metrics.accuracy_score(target_val, y_pred)
scores = cross_val_score(gnb, attrition_factored_column, target, scoring='accuracy', cv=10)
#78.49

# SVM
from sklearn import svm
svmmodel = svm.LinearSVC()
svmmodel.fit(smote_train, smote_target)
predicted = svmmodel.predict(test)

metrics.confusion_matrix(target_val, predicted, labels=None, sample_weight=None)
metrics.accuracy_score(target_val, predicted)
scores = cross_val_score(svmmodel, attrition_factored_column, target, scoring='accuracy', cv=10)
#87.21


# Neural Networks
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)
# create model
model = Sequential()
model.add(Dense(48, input_dim=31, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(train), np.array(target_train), epochs=1000, batch_size=10)

predictions = model.predict(np.array(test))
predictions = [round(x[0]) for x in predictions]
metrics.confusion_matrix(target_val, predictions, labels=None, sample_weight=None)
metrics.accuracy_score(target_val, predictions)
#84.239

# Feature Importances

svm_feature_importances = np.std(attrition_factored_column)*svmmodel.coef_.flatten()

trace = go.Scatter(
    y = svm_feature_importances,
    x = attrition_factored_column.columns.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 13,
        #size= rf.feature_importances_,
        #color = np.random.randn(500), #set color equal to a variable
        color = svm_feature_importances,
        colorscale='Portland',
        showscale=True
    ),
    text = attrition_factored_column.columns.values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)

plotly.offline.plot({"data":data, "layout":layout})