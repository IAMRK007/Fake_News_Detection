#@author Rajnish
#Coded on 20/09/2019
#Copyright AnonymousRK

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas
import pandas as pd
import warnings
warnings.filterwarnings('always')
dataset1 = pd.read_csv('/Volumes/RAJNISH EXT/PycharmProjects/majorfinal/1.csv',encoding = 'utf-8')
X1 = dataset1.iloc[:,0].values

X2 = dataset1.iloc[:,1].values
print(X2)
labels, texts = [], []
x=len(X2)
for i in range(0,x):
    texts.append(X1[i])
    if X2[i] == 'T':
        labels.append(1)
    else:
        labels.append(0)

trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['text'] = trainDF['text'].values.astype('U')
trainDF['label'] = labels
print(trainDF.head())
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.f1_score(predictions, valid_y, average='weighted', labels=None)

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("Using The Naive Bayes, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Using The Naive Bayes, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Using The Naive Bayes, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Using The Naive Bayes, CharLevel Vectors: ", accuracy)
# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("Using The Linear Regression, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Using The Linear Regression, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Using The Linear Regression, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Using The Linear Regression, CharLevel Vectors: ", accuracy)
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print("Using The Support Vector Machine Model , Count Vectors: ", accuracy)
accuracy = train_model(svm.SVC(),xtrain_tfidf, train_y, xvalid_tfidf)
print("Using The Support Vector Machine Model , Word level Vectors: ", accuracy)
accuracy = train_model(svm.SVC(),xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Using The Support Vector Machine Model , Char-level Vectors: ", accuracy)
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Using The Support Vector Machine Model , N-Gram Vectors: ", accuracy)