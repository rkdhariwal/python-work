import pandas as pd
df=pd.read_csv("/Users/raman/Desktop/MISM 6212/titanic_data-2.csv")

#Data Cleaning 
#drop the columns with missing values
df=df.drop(['Cabin', 'Name', 'Ticket', 'Age', 'Embarked',"PassengerId"], axis=1)


#convert to dummy variables 
df1= pd.get_dummies (df, drop_first = True)

#separate into x and y variables 
x=df1[["Pclass","SibSp", "Parch", "Fare", "Sex_male"]]

y=df1["Survived"]


#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

## build a decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

# evaluate the tree  
from sklearn.metrics import f1_score 

## training 
y_pred_train=dt.predict(x_train)
print ("training score is" , f1_score(y_train, y_pred_train))
#training score is 0.9002320185614849

#testing 
y_pred_test=dt.predict(x_test)
print ("testing score is" , f1_score(y_test, y_pred_test))
#testing score is 0.6981132075471699

##plot the tree 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()

##max depth 
dt.tree_.max_depth

#evaluate the model
from sklearn.metrics import f1_score,recall_score, precision_score
f1_score(y_test,y_pred_test)

# Classification report showing precision, recall, F-score
print ("Recall is", recall_score(y_test,y_pred_test))       
print ("F1 is", f1_score(y_test,y_pred_test))
print ("Precision is", precision_score(y_test,y_pred_test))



'''Recall is 0.6434782608695652
F1 is 0.6981132075471699
Precision is 0.7628865979381443'''


#Checking to see which Decision tree or random forest performed better

#random forest  
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)
rfc.fit(x_train,y_train)

## training 
y_pred_train=rfc.predict(x_train)
print ("training score is" , f1_score(y_train, y_pred_train))
#training score is 0.9006928406466513

#testing 
y_pred_test=rfc.predict(x_test)
print ("testing score is" , f1_score(y_test, y_pred_test))
#testing score is 0.7047619047619047

#evaluate the model
from sklearn.metrics import f1_score,recall_score, precision_score
f1_score(y_test,y_pred_test)

print ("Recall is", recall_score(y_test,y_pred_test))       
print ("F1 is", f1_score(y_test,y_pred_test))
print ("Precision is", precision_score(y_test,y_pred_test))


'''Recall is 0.6434782608695652
F1 is 0.7047619047619047
Precision is 0.7789473684210526'''

#Random forest model is better because the F1 score and the Precision score both improved 
#in random forest compared to the decision tree

#Tune the hyper-parameters of the decision tree and random forest model. 
#See how the performance of the tuned models compare with un-tuned models

## grid search to improve the decision tree 
parameter_grid = {"max_depth": range (2,16), "min_samples_split": range(2,6)}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt, parameter_grid, verbose=3, scoring="f1")

## train 
grid.fit(x_train,y_train)

## best parameters 
grid.best_params_


###using the parameters 
dt=DecisionTreeClassifier(max_depth=3, min_samples_split=2)
dt.fit(x_train,y_train)


## evaluate the tree now 
from sklearn.metrics import f1_score 

## training 
y_pred_train=dt.predict(x_train)
print ("training score is" , f1_score(y_train, y_pred_train))
#training score is 0.7500000000000001

#testing 
y_pred_test=dt.predict(x_test)
print ("testing score is" , f1_score(y_test, y_pred_test))
#testing score is 0.7053140096618357

#evaluate the model
from sklearn.metrics import f1_score,recall_score, precision_score
f1_score(y_test,y_pred_test)

print ("Recall is", recall_score(y_test,y_pred_test))       
print ("F1 is", f1_score(y_test,y_pred_test))
print ("Precision is", precision_score(y_test,y_pred_test))

#old - untuned decision tree
'''Recall is 0.6434782608695652
F1 is 0.6981132075471699
Precision is 0.7628865979381443'''

#new and tuned model decision tree
'''Recall is 0.6347826086956522
F1 is 0.7053140096618357
Precision is 0.7934782608695652'''

#The tuned decision tree model has improved F1 and Precision score therefore, the model has improved after tuning 

#tuning random forests to see if the model improves
parameter_grid = {"max_depth": range (2,16), "min_samples_split": range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(rfc, parameter_grid, verbose=3, scoring="f1")

## train 
grid.fit(x_train,y_train)

## best parameters 
grid.best_params_

#random forest with optimized parameters
rfc=RandomForestClassifier(n_estimators=500,max_depth=5,min_samples_split=2)
rfc.fit(x_train,y_train)

## training 
y_pred_train=rfc.predict(x_train)
print ("training score is" , f1_score(y_train, y_pred_train))
#training score is 0.7838479809976246

#testing 
y_pred_test=rfc.predict(x_test)
print ("testing score is" , f1_score(y_test, y_pred_test))
#testing score is 0.7093596059113301

#evaluate the model
from sklearn.metrics import f1_score,recall_score, precision_score
f1_score(y_test,y_pred_test)

print ("Recall is", recall_score(y_test,y_pred_test))       
print ("F1 is", f1_score(y_test,y_pred_test))
print ("Precision is", precision_score(y_test,y_pred_test))

#old untuned random forest model 
'''Recall is 0.6434782608695652
F1 is 0.7047619047619047
Precision is 0.7789473684210526'''

#new and tuned random forest model 
'''Recall is 0.6260869565217392
F1 is 0.7093596059113301
Precision is 0.8181818181818182'''

#After tuning the random forest, the F1 and precision increased and improved the model.
#overall the random forest model is still better than the decision tree model even after tuning.

##plot the tree 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()
































































