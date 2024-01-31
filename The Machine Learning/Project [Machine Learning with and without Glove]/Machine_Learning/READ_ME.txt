This folder contains the full project of the Machine Learning to create the models and calculate the accuracy and prediction of each based on two corpus.

Structure of the folder :
- "corpus-text-1m600.txt" : contains the corpus containing one million and six hundreds messages
- "corpus-text-medium.txt" : contains the corpus containing less than a thousands of messages
- "Data-to-be-filled-no_glove.txt" : file containing the vectors of the messages for the "corpus-text-medium.txt"
- "Data-to-be-filled-possible-terrorism-with_glove.txt" : file containing the vectors of the message sorted from the "corpus-text-1m600.txt"
- "GloVe_library.txt" : file containing the Glove library with fixed vectors for each words
- "M_L_L_Model.pkl","M_L_L_N_D_Model.pkl","M_L_R_Model.pkl","M_L_R_N_D_Model.pkl" : the four models created and saved  being [Model linear for "corpus-text-medium.txt", Model linear for
                                                                                    "corpus-text-1m600.txt", Model kernel for "corpus-text-medium.txt" , Model kernel for "corpus-text-1m600.txt"]
- "Machine_Learning_Main.py" : main implementation of the Machine Learning for the corpus "corpus-text-medium.txt"
- "Pre_Processing_Machine_Learning.py" : implementation of the pre_processing step for the messages
- "Machine_Learning_Domain_Adaptation_Model_Creation.py" : implementation to get the accuracy and prediction of the models for the corpus "corpus-text-1m600.txt"
- "Machine_Learning_Domain_Adaptation.py" : main implementation of the Machine Learning to create the models for the corpus "corpus-text-1m600.txt"

How to use the applications :
To create the two models using the Machine Learning with the Support Vector Machine for the corpus "corpus-text-medium.txt" without Glove, you need to launch "Machine_Learning_Main.py". 
It will create the two models and automatically display the accuracy and prediction for both of them.

To create the two models using the Machine Learning with the Support Vector Machine for the corpus "corpus-1m600.txt" with Glove, you need to launch "Machine_Learning_Domain_Adaptation.py".
Which will create the two models. If you want to know the accuracy and prediction for both of them, you will need to launch "Machine_Learning_Domain_Adaptation_Model_Creation.py"

Remark : The application may take some time to start, please be patient.

