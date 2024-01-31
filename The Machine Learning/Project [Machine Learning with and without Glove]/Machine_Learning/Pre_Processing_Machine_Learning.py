import re,nltk

def pre_processing(message):
    #print("Pre-processing of the message:", message)
    dict2 = list()
    dict1 = nltk.sent_tokenize(message)  #Split the tweet in sentences#
    #print("Message tokenized by sentences:", dict1)
#######################################################
    for i in dict1:
        a = ' '.join(re.sub("(@)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i).split())  #Delete all useless word such as "@"#
        dict2.append(a)
    #print("Message without special characters:", dict2)
####################################################
    dict1 = []
    for i in dict2:
        a = nltk.word_tokenize(i)                              #Split each sentences in words#
        dict1.append(a)
    #print("Message tokenized as words:", dict1)
##################################################
    filter_words = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    for i in dict1:                                           # For each word, will discard the stop_words (common)#
        for j in i:
            if j not in stop_words:
                filter_words.append(j)
    #print("Filtered words without English stopwords:", filter_words)
###################################################
    dict2 = []
    for i in filter_words:  # Lemmatization#
        a = nltk.stem.WordNetLemmatizer().lemmatize(i)
       # b = nltk.stem.PorterStemmer().stem(a)
        dict2.append(a)
    #print("Lemmatization of each word:", dict2)
##################################################
    return dict2

