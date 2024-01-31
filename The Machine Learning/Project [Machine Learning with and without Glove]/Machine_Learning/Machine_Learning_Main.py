import csv
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(font_scale=1.2)

from time import sleep
from Pre_Processing_Machine_Learning import pre_processing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

print("################################################################################################################################################")
print("# Project : Machine Learning on data words using the Support Vector Machine [SVM] technique with no GlOve.                                     #")
print("# Author : ACHTATOU SEAN                                                                                                                       #")
print("#                                                                                                                                              #")
print("# University of Luxembourg, Bachelor In Computer Science.                                                                                      #")
print("# All rights reserved.                                                                                                                         #")
print("################################################################################################################################################")
sleep(5)
#######################################################################################################################
print("Please wait. The messages contents and sentiment are being read from the csv_file.")
sleep(2)
data_message = [] #Create set in which we put the messages#
data_sentiment = [] #Set in which we put the sentiment#
with open("corpus-text-medium.txt") as csv_file:  #Read the file#
    csv_reader = csv.reader(csv_file, delimiter = ",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:                                              #Read the data(tweet,sentiment) of the csv file and put them in lists#
            line_count += 1
            continue
        elif row == []:
            break
        else:
            data_message.append(row[4])#Add message#
            data_sentiment.append(row[1]) #Add sentiment#
    csv_file.close()
#######################################################################################################################
print("Contents of messages and sentiment read and put in set successfully.")
print("")
print("Please wait. The messages are being pre-processed.")
Set = []
for i in data_message:
    a = pre_processing(i.lower())                                          #Pre_process each tweet#
    b = nltk.FreqDist(a)                                                   #Calculate the freq of words for a tweet#
    features = dict()
    word_features = list(b.keys())
    word_features_values = list(b.values())
    n = 0
    for i in word_features:
        features[i] = word_features_values[n]
        n += 1
    print("Features words of the tweet with the numbers it appears :",features)
    for i in word_features:
        if i in Set:
            continue
        else:
            Set.append(i)
    print("Features for all the tweet for now :",Set)                                  #Show the number of features for the training data#
    print("_______________________________________________________________________")
print("The messages have been pre-processed successfully.")
########################################################################################################################
print("")
print("Please wait. The messages are being represented as a vector to be used in the Machine Learning model.")
sleep(2)
initialfeatures = []
for i in range(len(Set)):
    initialfeatures.append(0)                                            #Create the list of 0 according to the length of the features#
finalfeatures = initialfeatures.copy()

finaldict = []
m = 0
for i in data_message:
    m += 1
    n = -1
    initialfeatures = finalfeatures.copy()                               #For each tweet, see if a feature is in there and add one if True#
    for j in Set:
        n += 1
        if j in i.lower():
            initialfeatures[n] += 1
    finaldict.append(initialfeatures) #Show for each tweet what is the axes#
##################################################################################################################################################
print("")
print("The messages have been successfully represented into a vector.")
print("All the vectors are being put in a csv_file such as a matrix.")
sleep(2)
with open("Data-to-be-filled-no_glove.txt","w") as csv_file:
    fieldnames = ["Sentiment"]
    for i in Set:
        fieldnames.append(i)
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)               #Put all the data in a .csv_file#
    writer.writeheader()
    writer = csv.writer(csv_file)
    n = 0
    for i in data_message:
        a = [data_sentiment[n]]
        for j in finaldict[n]:
            a.append(j)
        writer.writerow(a)
        n += 1
csv_file.close()
print("Vectors successfully put in the csv_file.")
###################################################################################################################################################
print("")
print("Beginning of the construction of the models for the Machine Learning.")
sleep(2)
dataset = pd.read_csv("Data-to-be-filled-no_glove.txt")                                    #Represent all out data#
dataset.head()

features_words = dataset[Set].as_matrix()  #Put the features as a matrix#
type_label_initial = dataset["Sentiment"].as_matrix()
type_label = np.where(type_label_initial=="positive",0,1)

print("Division of a set of vectors for training and testing.")
X_train, X_test, Y_train, Y_test = train_test_split(features_words,type_label , test_size = 0.2 ,random_state= 0)
sleep(2)

print("Creation of the linear model and the rbf model.")
model = svm.SVC(kernel="linear",C = 10, class_weight= "balanced")                            #model for kernel = linear : C = 10#         #Create the model#
model2 = svm.SVC(kernel="rbf", C = 2, class_weight = "balanced", gamma = 0.01)               #mode for kernel = rbf : C = 2 , gamma = 0.01#
sleep(2)

print("Input of the training vectors in each model.")
model.fit(X_train,Y_train)
model2.fit(X_train,Y_train)                                                  #Fit the models with the training data#
sleep(2)

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
machine_learning_model_linear = "M_L_L_Model.pkl"
machine_learning_model_rbf = "M_L_R_Model.pkl"
joblib.dump(model,machine_learning_model_linear)  #Save the model#
joblib.dump(model2,machine_learning_model_rbf)

input("Press Enter key to quit...")
















