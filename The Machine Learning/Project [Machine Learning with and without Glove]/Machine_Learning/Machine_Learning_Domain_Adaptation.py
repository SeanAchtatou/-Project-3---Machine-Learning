import csv
import sys
import pandas as pd

from time import sleep
from Pre_Processing_Machine_Learning import pre_processing

########################################################################################################################
print("################################################################################################################################################")
print("# Project : Machine Learning on big set of data words using the Support Vector Machine [SVM] technique with the GlOve.                         #")
print("# Author : ACHTATOU SEAN                                                                                                                       #")
print("# Description : Creation of Machine Learning models on a random domain of sentences [1m600]. Selection according to terrorism behaviour.       #")
print("# Part : 1/2                                                                                                                                   #")
print("#                                                                                                                                              #")
print("# University of Luxembourg, Bachelor In Computer Science.                                                                                      #")
print("# All rights reserved.                                                                                                                         #")
print("################################################################################################################################################")
sleep(5)
print("")
########################################################################################################################

wordstocheck = ["bomb","terrorism","explosion","gun","fear","scary","death","shot","dead","terrorist","arab","arabics","die","weapon","destruction","beard","vest","isis",
                "religion","allah","kill","killer"]

print("Word related to terrorism which will be checked in the corpus of 1 million and 600 hundreds messages:",wordstocheck)

csv.field_size_limit(sys.maxsize)                            # Because we are working with huge data the csv_file might become overloaded, so we increase the max limit#

print("Please wait. The messages are being classified according to the terrorists words. It may take a while. Estimated time : 40 seconds. ")
########################################################################################################################
possibletreats = []
sentiment= []
with open("corpus-text-1m600.txt") as csv_file:
    csv_reader = csv.reader(csv_file,delimiter=",")                                    #Read the csv_file and look for messages with the wordstocheck words#
    for row in csv_reader:
        if row == []:
            break
        else:
            for i in wordstocheck:
                if i in row[5].lower():
                    for j in range(wordstocheck.index(i)+1,len(wordstocheck)):
                        if wordstocheck[j] in row[5].lower():
                            if row[5] in possibletreats:  # The corpus might have same sentence leading to a different sentiment, we need to delete those#
                                break
                            else:
                                possibletreats.append(row[5])
                                if row[0] == "4":
                                    sentiment.append(1)
                                else:
                                    sentiment.append(0)
                            break
                    break

    csv_file.close()
print("The messages have been classified.")
print("Messages containing terrorist behaviour :",len(possibletreats),possibletreats)
print("Number of possibles messages containing terrorist behaviour : ",len(possibletreats))
print("Sentiment for each messages [0 = positive / 1 = negative] :",sentiment)
print("")
print("Please wait. The possibles terrorist messages are being given vector according to the GlOve library. It may take a while. Estimated time : 10 minutes. ")
########################################################################################################################
Set = []
Final_glOve_set = []
counting = 0
for i in possibletreats:
    counting += 1
    GlOve_set = [0] * 50
    a = pre_processing(i.lower())                                                      #Pre_process each messages,glove it and input it in a csv_file#
    for j in a:
        with open("GloVe_library.txt") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            for row in csv_reader:
                if row[0] == j:                                                             # glove the words for each sentence#
                    for k in range(50):
                        GlOve_set[k] += float(row[k+1])
                    break
        csv_file.close()
    Final_glOve_set.append(GlOve_set)
    for i in a:
        if i in Set:
            continue
        else:
            Set.append(i)
    #print("Words for now : ", Set)
print("_________________________________________________________________________________________________________________________________________")
print("Vectors set for each messages with potential terrorism relation.")
print("All words : ",len(Set),Set)
########################################################################################################################
Lineset = []
for i in range(51):
    Lineset.append(i)

n = 0
with open("Data-to-be-filled-possible-terrorism-with_glove.txt", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(Lineset)
    for i in range(len(possibletreats)):
        x = [sentiment[n]]                                                                               # input it in the data "Data to be filled..."#
        for j in Final_glOve_set[n]:
            x.append(j)
        writer.writerow(x)
        n += 1

csv_file.close()

########################################################################################################################

newdataset = pd.read_csv("Data-to-be-filled-possible-terrorism-with_glove.txt")                           #Put in matrix the data#
features_words = newdataset.as_matrix()

input("Press Enter key to quit...")


                                                                                                     #Fit the data in the model and prediction with the linear and rbf mode#
