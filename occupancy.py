##Step1:- Load the data

import numpy as np
import pandas as pd

train=pd.read_csv("C:/Users/Avinash/Desktop/occupancy_data (1)/datatraining.txt")
train['date']=pd.to_datetime(train['date']) #convert date time into python datetime
train.reset_index(drop=True,inplace=True)

Validation=pd.read_csv("C:/Users/Avinash/Desktop/occupancy_data (1)/datatest.txt")
Validation['date']=pd.to_datetime(Validation['date']) #convert date time into python datetime
Validation.reset_index(drop=True,inplace=True)

test=pd.read_csv("C:/Users/Avinash/Desktop/occupancy_data (1)/datatest2.txt")
test['date']=pd.to_datetime(test['date']) #convert date time into python datetime
test.reset_index(drop=True,inplace=True)


##Step2:- Cocatenate
occupancy_data=pd.concat([train,test,Validation])
occupancy_data.head(10)
occupancy_data.shape
print(occupancy_data.info())
occupancy_data.describe()


###Step 3: EDA & Visualisation
import  matplotlib.pyplot as plt
import seaborn as sns; 

plt.figure(figsize=(10,10))
plt.title('Correlation table heatmap')
ax=sns.heatmap(occupancy_data.corr(),annot=True,linewidths=.2)


sns.set(style="darkgrid")
plt.title("Occupancy Distribution", fontdict={'fontsize':18})
ax = sns.countplot(x="Occupancy", data=occupancy_data)


from plotnine import ggplot, aes, geom_line

(
    ggplot(occupancy_data)  # What data to use
    + aes(x="date", y="Temperature")  # What variable to use
    + geom_line()  # Geometric object to use for drawing
)


(
    ggplot(occupancy_data)  # What data to use
    + aes(x="date", y="Humidity")  # What variable to use
    + geom_line()  # Geometric object to use for drawing
)


(
    ggplot(occupancy_data)  # What data to use
    + aes(x="date", y="Light")  # What variable to use
    + geom_line()  # Geometric object to use for drawing
)


(
    ggplot(occupancy_data)  # What data to use
    + aes(x="date", y="CO2")  # What variable to use
    + geom_line()  # Geometric object to use for drawing
)


#Our data is unbalanced, so we need to find another relations 
#between features to strengthen our predictions.
#I have a question at this point, is there any relation between occupancy
# and the hour of the day? Let's look into it.

hours_1 = []
hours_0 = []
for date in occupancy_data[occupancy_data['Occupancy'] == 1]['date']:
    hours_1.append(date.hour)
for date in occupancy_data[occupancy_data['Occupancy'] == 0]['date']:
    hours_0.append(date.hour)
    
plt.figure(figsize=(8,8))
plt.xlabel("Time")
plt.ylabel("Density")
ax = sns.distplot(hours_1)
ax = sns.distplot(hours_0)

#From above histogram, what can you say? Between 07:00 and 18:00 
#there are occupants in the environment or not.
#But the time come to non-working hours, then we can absolutely 
#say that there is no occupant. With this information, 
#I will create a new feature from date column as day period.

occupancy_data['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in occupancy_data['date']]
occupancy_data.sample(10)


#07:00 - 18:00 working hour (labeled as 1)
#rest of the day non-working hour (labeled as 0)

###Step4:-Model Building

from sklearn.model_selection import train_test_split
# create dataset
X = occupancy_data.iloc[:,0:8] # :colon is used when all rows
                        # 0:2 for 0th,1st col
y = occupancy_data.iloc[:,6]

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,stratify=y,random_state=2020)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


X_train=X_train.drop(columns=['date','Occupancy'],axis=1)
X_test=X_test.drop(columns=['date','Occupancy'],axis=1)

##1 Logistic model

##Logistic Regression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_model = LogisticRegression()
# fit the model on the training set
logit_model.fit(X_train, y_train)
# predict the test set
yhat = logit_model.predict(X_test)
yhat1=logit_model.predict(X_train)
# evaluate model skill
score = accuracy_score(y_test, yhat)
score1=accuracy_score(y_train,yhat1)
print(score)
print(score1)

####HyperParameter tuning for Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# define models and parameters
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=c_values)
model_params={
    'logistic':{
        'model':LogisticRegression(),
        'params':grid
        }
    
    }

scores = []
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
logistic_df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
logistic_df


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
print(confusion_matrix(y_test, yhat))
print(confusion_matrix(y_train,yhat1))
print(classification_report(y_test, yhat))
print(accuracy_score(y_test, yhat))


plt.title("Logistic Confusion Matrix for Test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, yhat), annot=True, fmt="d")

plt.title("Logistic Confusion Matrix for Train Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_train, yhat1), annot=True, fmt="d")


##2. Ridge Classifier


from sklearn.linear_model import RidgeClassifier
####HyperParameter tuning for Ridge Classifier

alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Ridge_grid = dict(alpha=alpha)
Ridge_params={
    'Ridge':{
        'model':RidgeClassifier(),
        'Ridge_params':Ridge_grid
        }
    
    }

Ridge_scores = []

for model_name, mp in Ridge_params.items():
    clf =  GridSearchCV(mp['model'], mp['Ridge_params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    Ridge_scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
Ridge_df = pd.DataFrame(Ridge_scores,columns=['model','best_score','best_params'])
Ridge_df


model = RidgeClassifier()
# fit the model on the training set
model.fit(X_train, y_train)
# predict the test set
yhat = model.predict(X_test)
yhat1=model.predict(X_train)
# evaluate model skill
score = accuracy_score(y_test, yhat)
score1=accuracy_score(y_train, yhat1)
print(score)
print(score1)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
print(confusion_matrix(y_test, yhat))
print(classification_report(y_test, yhat))
print(accuracy_score(y_test, yhat))



plt.title("Ridge Classifier Confusion Matrix for Test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, yhat), annot=True, fmt="d")

plt.title("Ridge Classifier Confusion Matrix for Train Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_train, yhat1), annot=True, fmt="d")


#  3. KNN Algorithm 
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
## KNN with Manhattan and Euclidean
# parameter-tuning for knn
n_neighbors =range(1,200,2)
weights = ['uniform', 'distance']
metric= ['euclidean', 'manhattan']
Knn_grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

KNN_params={
    'K_Neighbours':{
        'model':KNeighborsClassifier(),
        'params':Knn_grid
        }
    
    }

Knn_scores = []

for model_name, mp in KNN_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    Knn_scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
Knn_df = pd.DataFrame(Knn_scores,columns=['model','best_score','best_params'])
Knn_df

from sklearn.metrics import confusion_matrix
knn_model = KNeighborsClassifier(n_neighbors=9,metric='manhattan',weights='distance')
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
y_pred1=knn_model.predict(X_train)
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_train, y_pred1))


plt.title("KNN Confusion Matrix for testing Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

plt.title("KNN Confusion Matrix for training Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_train, y_pred1), annot=True, fmt="d")


###4. SVM (Support Vector machine)
from sklearn.svm import SVC
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
SVM_grid = dict(kernel=kernel,C=C,gamma=gamma)

SVM_params={
    'SVM':{
        'model':SVC(),
        'params':SVM_grid
        }
    
    }
SVM_scores = []
for model_name, mp in SVM_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    SVM_scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
SVM_df = pd.DataFrame(SVM_scores,columns=['model','best_score','best_params'])
SVM_df

svm_model = SVC(C=50,gamma='scale',kernel='rbf')
svm_model.fit(X_train, y_train)
print("Accuracy for SVM on validation data: {}%".format(round((svm_model.score(X_test, y_test)*100),2)))
print("Accuracy for SVM on validation data: {}%".format(round((svm_model.score(X_train, y_train)*100),2)))
y_pred = svm_model.predict(X_test)
y_pred1=svm_model.predict(X_train)
plt.title("SVM Confusion Matrix for test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("SVM Confusion Matrix for train Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_train, y_pred1), annot=True, fmt="d")


###Random Forest Modell
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
##Hyper Parameter Tuning for Random Forest
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
rfgrid = dict(n_estimators=n_estimators,max_features=max_features)

rfparams={
    'RandomForest':{
        'model':RandomForestClassifier(random_state=2020,oob_score=True),
        'params':rfgrid
        }
    
    }

rfscores = []

for model_name, mp in rfparams.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    rfscores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
rf_df = pd.DataFrame(rfscores,columns=['model','best_score','best_params'])
rf_df



model_rf = RandomForestClassifier(random_state=2020,max_features='sqrt',
                                  n_estimators=100,oob_score=True)
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)
y_pred1=model_rf.predict(X_train)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_train, y_pred1))
plt.title("Random Forest Confusion Matrix for Test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

plt.title("Random Forest Confusion Matrix for Train Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_train, y_pred1), annot=True, fmt="d")


###6. Bagged Decision Trees (Bagging)
from sklearn.ensemble import BaggingClassifier
n_estimators = [10, 100, 1000]
# define grid search
bagging_grid = dict(n_estimators=n_estimators)

baggingparams={
    'BaggingClassifier':{
        'model':BaggingClassifier(random_state=2020),
        'params':bagging_grid
        }
    
    }

bg_scores = []

for model_name, mp in baggingparams.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    bg_scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
bg_df = pd.DataFrame(bg_scores,columns=['model','best_score','best_params'])
bg_df


model =BaggingClassifier(n_estimators=10,random_state=2020)
# fit the model on the training set
model.fit(X_train, y_train)
# predict the test set
yhat1=model.predict(X_train)
yhat = model.predict(X_test)
# evaluate model skill
score = accuracy_score(y_test, yhat)
print(score)

score1 = accuracy_score(y_train, yhat1)
print(score1)

print(confusion_matrix(y_test, yhat))
print(classification_report(y_test, yhat))
print(accuracy_score(y_test, yhat))


plt.title("Bagging Confusion Matrix for Test Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_test, yhat), annot=True, fmt="d")

plt.title("Bagging Confusion Matrix for Train Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_train, yhat1), annot=True, fmt="d")

Optimalmodel_params = {
    'svm': {
        'model':SVC(),
        'params':SVM_grid        
    },
    'random_forest': {
        'model':RandomForestClassifier(random_state=2020,oob_score=True),
        'params':rfgrid
        
    },
    'logistic_regression' : {
        'model':LogisticRegression(),
        'params':grid
        
        
    },
    'RidgeClassifier':{
        
        'model':RidgeClassifier(),
        'params':Ridge_grid
        
    },
    'Kneighbours': {
        
        'model':KNeighborsClassifier(),
        'params':Knn_grid
    
    },
    
    'BaggingClassifier': {
        
        'model':BaggingClassifier(random_state=2020),
        'params':bagging_grid
    
    }
}


optimal_scores=[]
for model_name, mp in Optimalmodel_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_test,y_test)
    optimal_scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
Optimal_df = pd.DataFrame(optimal_scores,columns=['model','best_score','best_params'])
Optimal_df


# Saving model to disk

import pickle
pickle.dump(model_rf,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

model.predict([[20.7,22.5,0,454.5,0.00339,0]])


occupancy_data['Occupancy'].value_counts()

df_majority = occupancy_data[occupancy_data.Occupancy==0]
df_minority = occupancy_data[occupancy_data.Occupancy==1]

from sklearn.utils import resample
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=4750,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.Occupancy.value_counts()

y1=df_downsampled.Occupancy
x1=df_downsampled.drop(columns=['date','Occupancy'],axis=1)
x1.shape
y1.shape
clf_1 = LogisticRegression().fit(x1, y1)
pred_y_1 = clf_1.predict(x1)
print(np.unique( pred_y_1 ) )
print(accuracy_score(y1, pred_y_1) )


from sklearn.utils import resample
# Downsample majority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=15810,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
 
# Display new class counts
df_upsampled.Occupancy.value_counts()

y1=df_upsampled.Occupancy
x1=df_upsampled.drop(columns=['date','Occupancy'],axis=1)
x1.shape
y1.shape
clf_1 = LogisticRegression().fit(x1, y1)
pred_y_1 = clf_1.predict(x1)
print(np.unique( pred_y_1 ) )
print(accuracy_score(y1, pred_y_1) )

