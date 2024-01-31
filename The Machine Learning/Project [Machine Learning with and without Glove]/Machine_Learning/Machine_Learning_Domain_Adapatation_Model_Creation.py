import pandas as pd

from time import sleep
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

########################################################################################################################
print("################################################################################################################################################")
print("# Project : Machine Learning on big set of data words using the Support Vector Machine [SVM] technique with the GlOve.                         #")
print("# Author : ACHTATOU SEAN                                                                                                                       #")
print("# Description : Creation of Machine Learning models on a random domain of sentences [1m600]. Selection according to terrorism behaviour.       #")
print("# Part : 2/2                                                                                                                                   #")
print("#                                                                                                                                              #")
print("# University of Luxembourg, Bachelor In Computer Science.                                                                                      #")
print("# All rights reserved.                                                                                                                         #")
print("################################################################################################################################################")
sleep(5)
print("")
########################################################################################################################
print("Beginning of the construction of the models for the Machine Learning.")
Set = []
a = 1
for i in range(50):
    Set.append(str(a))
    a += 1

newdataset = pd.read_csv("Data-to-be-filled-possible-terrorism-with_glove.txt")                           #Put in matrix the data#
newdataset.head()
features_words = newdataset[Set].as_matrix()
type_label = newdataset["0"].as_matrix()

print("Division of a set of vectors for training and testing.")
X_train, X_test, Y_train, Y_test = train_test_split(features_words,type_label , test_size = 0.3 ,random_state= 0)

print("Creation of the linear model and the rbf model.")
model = svm.SVC(kernel="linear", C=5, class_weight="balanced")                                   #model for kernel = linear : C = 0.7#         #Create the model#
model2 = svm.SVC(kernel="rbf", C=3, class_weight = "balanced", gamma = 0.01)               #mode for kernel = rbf : C = 6 , gamma = 0.05#

print("Input of the training vectors in each model.")
model.fit(X_train,Y_train)
model2.fit(X_train,Y_train)                                                  #Fit the models with the training data#

score = model.score(X_test,Y_test)                                     #Accuracy and prediction for both models#
score2 = model2.score(X_test,Y_test)
PredictionY = model.predict(X_test)
PredictionY2 = model2.predict(X_test)

print("")
print("Table to test the prediction of the linear model and the real results:")

n = 0
for i in PredictionY:
    print("Prediction linear model:",i,"/","Prediction rbf model:",PredictionY2[n],"/","Real value:",Y_test[n])          #Show the table of prediction for both model#
    n += 1
print("")
print("Probability to get a good result for linear model:" ,score)
print("Probability to get a good result for rbf model:" ,score2)
#############################################################################################################################################
n = -1
falsepositivelinear = 0
falsenegativelinear = 0
falsepositiverbf = 0
falsenegativerbf = 0

for i in PredictionY:
    n += 1
    if i != Y_test[n]:
        if i == 0:
            falsepositivelinear += 1                                                             #Probability to have T-P/T-N/F-N/F-P [cross-validation]#
        else:
            falsenegativelinear += 1
    elif PredictionY2[n] != Y_test[n]:
        if PredictionY2[n] == 0:
            falsepositiverbf += 1
        else:
            falsenegativerbf += 1
    else:
        continue

print("")
print("Cross-validation:")
print("")
print("True positive probability for linear model:" , 1 - falsepositivelinear/len(PredictionY))
print("False positive probability for linear model:" , falsepositivelinear/len(PredictionY))
print("True negative probability for linear model:" , 1 - falsenegativelinear/len(PredictionY))
print("False negative probability for linear model:" , falsenegativelinear/len(PredictionY))
print("")
print("True positive probability for rbf model:" , 1 - falsepositiverbf/len(PredictionY))
print("False positive probability for rbf model:" , falsepositiverbf/len(PredictionY))
print("True negative probability for rbf model:" , 1 - falsenegativerbf/len(PredictionY))
print("False negative probability for rbf model:" , falsenegativerbf/len(PredictionY))

######################################################################################################################################################
machine_learning_model_linear_new_domain = "M_L_L_N_D_Model.pkl"
machine_learning_model_rbf_new_domain = "M_L_R_N_D_Model.pkl"
joblib.dump(model,machine_learning_model_linear_new_domain)  #Save the model#
joblib.dump(model2,machine_learning_model_rbf_new_domain)

input("Press Enter key to quit...")
