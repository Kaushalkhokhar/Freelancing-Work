import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plotting_IRIS_data():
    data = sns.load_dataset('iris')
    data.head()

    plt.figure(figsize=(15,10))
    #plt.subplot()
    sns.pairplot(data,hue="species", markers=["o", "s", "D"])
    #plt.scatter(data.iloc[:,:-1],y='species')
    #plt.title("Iris Dataset", loc='center')
    plt.show()

def IBM_data_preparation():
    global X, y
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    df.pop('EmployeeCount')
    df.pop('EmployeeNumber')
    df.pop('Over18')
    df.pop('StandardHours')
    df.shape

    y = df['Attrition']
    X = df
    X.pop('Attrition')

    from sklearn.preprocessing import LabelBinarizer
    le = LabelBinarizer()
    y = le.fit_transform(y)
    y.shape

    colm = {'BusinessTravel': 'BusinessTravel_new',
            'Department': 'Department_new', 
            'EducationField':'EducationField_new',
            'Gender':'Gender_new',
            'JobRole':'JobRole_new',
            'MaritalStatus':'MaritalStatus_new',
            'OverTime':'OverTime_new'}

    data_frame = pd.DataFrame({'0': range(1470)})
    data_frame = data_frame.astype('uint8')
    #print(data_frame.shape)
    for i in colm.keys():
        #print(i)
        temp = pd.get_dummies(df[str(i)], prefix = str(i))
        #print(temp.shape)
        data_frame = pd.concat([data_frame, temp], axis = 1)
        #ind_BusinessTravel = pd.get_dummies(df[str(i)], prefix = str(i))
    df1 = data_frame.iloc[:,1:]
   
    df1 = pd.concat([df1, df.select_dtypes('int64')], axis = 1 )
    X = df1

IBM_data_preparation()

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import xgboost as xgb

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# To print score 
def print_score(clf, X_train, X_test, y_train, y_test, train = True):
    '''
    To Print the accuracy score, Precision and recall score with confusion metrcis
    '''
        
    if train:
        '''
        Print the score of train datasets
        '''
        print("\n Report of {} \n".format(clf))
        print("\n @ Train datasets\n")
        
        print("Accuracy score: \t{0:.4f}".format(accuracy_score(y_train, clf.predict(X_train))))
        print("classification_report: \n {} \n".format(classification_report(y_train, clf.predict(X_train))))
        print("confusion matrics: \n {} \n".format(confusion_matrix(y_train, clf.predict(X_train))))
        
        res = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy')
        print("Average Accuracy using 10-fold CV: \t {0:.4f} \n SD using 10-fold CV: \t\t\t{1:.4f}".format(np.mean(res), np.std(res)))
        
    elif train == False:
        '''
        Print the score of test sets
        '''
        print("\n @ Test datasets\n")
        
        print("Accuracy score: \t {0:.4f}".format(accuracy_score(y_test,clf.predict(X_test))))
        print("classification_report: \n {} \n".format(classification_report(y_test, clf.predict(X_test))))
        print("confusion matrics: \n {} \n".format(confusion_matrix(y_test, clf.predict(X_test))))
        


# Define classifier
#clf = DecisionTreeClassifier()
#bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=100, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42)
xgb_clf = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, n_jobs=-1, gamma=0, random_state=42)

# Fittinng the data
xgb_clf.fit(X_train, y_train.flatten())

print_score(xgb_clf, X_train, X_test, y_train, y_test, train = True)
print_score(xgb_clf, X_train, X_test, y_train, y_test, train = False)