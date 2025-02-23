'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import sentiment_read_subjectivity
import os
import sys
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
import re
#import sentiment_read_subjectivity

stopwords = nltk.corpus.stopwords.words('english')
newstopwords = [word for word in stopwords if word not in ['not', 'no', 'can','don', 't']]



def pre_processing(document):
  # "Pre_processing_documents"  
  # "create list of lower case words"
  word_list = re.split('\s+', document.lower())

  punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
  word_list = [punctuation.sub("", word) for word in word_list] 
  final_word_list = []
  for word in word_list:
    if word not in newstopwords:
      final_word_list.append(word)
  line = " ".join(final_word_list)
  return line 

def get_words_from_docs(docs):
  all_words = []
  for (words, sentiment) in docs:
    # more than 3 length
    possible_words = [x for x in words if len(x) >= 3]
    all_words.extend(possible_words)
  return all_words

def get_words_from_docs_usual(docs):
  all_words = []
  for (words, sentiment) in docs:
    all_words.extend(words)
  return all_words  

# get all words from tokens
def get_words_from_test_dataset(lines):
  all_words = []
  for id,words in lines:
    all_words.extend(words)
  return all_words


def write_feature_sets(featuresets, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # write this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
          featureline += str(featureset[0][key]) + ','
        if featureset[1] == 0:
          featureline += str("neg")
        elif featureset[1] == 1:
          featureline += str("sneg")
        elif featureset[1] == 2:
          featureline += str("neu")
        elif featureset[1] == 3:
          featureline += str("spos")
        elif featureset[1] == 4:
          featureline += str("pos")
        # write each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()


def get_word_features(wordlist):
  wordlist = nltk.FreqDist(wordlist)
  word_features = [w for (w, c) in wordlist.most_common(200)] 
  return word_features    


def usual_features(document, word_features):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  return features


def bigram_document_features(document, word_features,bigram_features):
  document_words = set(document)
  document_bigrams = nltk.bigrams(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  for bigram in bigram_features:
    features['bigram({} {})'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
  return features

def get_bigram_features(tokens):
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(tokens,window_size=3)
  bigram_features = finder.nbest(bigram_measures.chi_sq, 3000)
  return bigram_features[:500]

def trigram_document_features(document, word_features,trigram_features):
  document_words = set(document)
  document_trigrams = nltk.trigrams(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  for trigram in trigram_features:
    #print(trigram)
    features['trigram({} {} {})'.format(trigram[0], trigram[1], trigram[2])] = (trigram in document_trigrams)    
  return features

def get_trigram_features(tokens):
  trigram_measures = nltk.collocations.TrigramAssocMeasures()
  finder = TrigramCollocationFinder.from_words(tokens,window_size=3)
  #finder.apply_freq_filter(6)
  trigram_features = finder.nbest(trigram_measures.chi_sq, 3000)
  return trigram_features[:500]

def readSubjectivity(path):
  flexicon = open(path, 'r')
  # initialize an empty dictionary
  sldict = { }
  for line in flexicon:
    fields = line.split()
    strength = fields[0].split("=")[1]
    word = fields[2].split("=")[1]
    posTag = fields[3].split("=")[1]
    stemmed = fields[4].split("=")[1]
    polarity = fields[5].split("=")[1]
    if (stemmed == 'y'):
      isStemmed = True
    else:
      isStemmed = False
    # put a dictionary entry with the word as the keyword
    #     and a list of the other values
    sldict[word] = [strength, posTag, isStemmed, polarity]
  return sldict

SLpath = "./SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"
SL = readSubjectivity(SLpath)
def SL_features(document, word_features, SL):
  document_words = set(document)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  # count variables for the 4 classes of subjectivity
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in document_words:
    if word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
      features['positivecount'] = weakPos + (2 * strongPos)
      features['negativecount'] = weakNeg + (2 * strongNeg)
  
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0      
  return features


negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather',
                 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

def NOT_features(document, word_features, negationwords):
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = False
    features['contains(NOT{})'.format(word)] = False
  # go through document words in order
  for i in range(0, len(document)):
    word = document[i]
    if ((i + 1) < len(document)) and (word in negationwords):
      i += 1
      features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
    else:
      if ((i + 3) < len(document)) and (word.endswith('n') and document[i+1] == "'" and document[i+2] == 't'):
        i += 3
        features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
      else:
        features['contains({})'.format(word)] = (word in word_features)
  return features


def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features


def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])

  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
  #for phrase in phraselist[:10]:
    #print (phrase)
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  phrasedocs_without = []
  # add all the phrases
  for phrase in phraselist:

    #without preprocessing
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs_without.append((tokens, int(phrase[1])))
    
    # with pre processing
    tokenizer = RegexpTokenizer(r'\w+')
    phrase[0] = pre_processing(phrase[0])
    tokens = tokenizer.tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
  
  # possibly filter tokens
  normaltokens = get_words_from_docs_usual(phrasedocs_without)
  preprocessedTokens = get_words_from_docs(phrasedocs)


  word_features = get_word_features(normaltokens)
  featuresets_without_preprocessing = [(usual_features(d, word_features), s) for (d, s) in phrasedocs_without]
  write_feature_sets(featuresets_without_preprocessing,"featuresets_without_preprocessing.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with normal features, without pre-processing steps : ")
  accuracy_calculation(featuresets_without_preprocessing)


  word_features = get_word_features(preprocessedTokens)

  featuresets = [(usual_features(d, word_features), s) for (d, s) in phrasedocs]
  write_feature_sets(featuresets,"featuresets.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with pre-processed features : ")
  accuracy_calculation(featuresets)

  
  SL_featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in phrasedocs]
  write_feature_sets(SL_featuresets,"features_SL.csv")
  #print SL_featuresets[0]
  print ("---------------------------------------------------")
  print ("Accuracy with SL_featuresets : ")
  accuracy_calculation(SL_featuresets)

  NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in phrasedocs]
  #print NOT_featuresets[0]
  write_feature_sets(SL_featuresets,"features_NOT.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with NOT_featuresets : ")
  accuracy_calculation(NOT_featuresets)

  POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in phrasedocs]
  #print NOT_featuresets[0]
  write_feature_sets(POS_featuresets,"features_POS.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with POS_featuresets : ")
  accuracy_calculation(POS_featuresets)

  bigram_features = get_bigram_features(preprocessedTokens)
  #print(bigram_features[0])
  bigram_featuresets = [(bigram_document_features(d, word_features,bigram_features), c) for (d, c) in phrasedocs]
  #print(bigram_featuresets[0])
  write_feature_sets(bigram_featuresets,"features_bigram.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with bigram featuresets : ")
  accuracy_calculation(bigram_featuresets)

  trigram_features = get_trigram_features(preprocessedTokens)
  #print (trigram_features[0])
  trigram_featuresets = [(trigram_document_features(d, word_features,trigram_features), c) for (d, c) in phrasedocs]
  #print (trigram_featuresets[0])
  write_feature_sets(bigram_featuresets,"features_trigram.csv")
  print ("---------------------------------------------------")
  print ("Accuracy with Trigram featuresets : ")
  accuracy_calculation(trigram_featuresets)

  features_combined = [(combined_document_features(d, word_features, SL_featuresets, bigram_featuresets), c) for (d, c) in phrasedocs]
  print("Accuracy with combined featuresets : ")
  accuracy_calculation(features_combined)


def accuracy_calculation(featuresets):
  print ("Training and testing a classifier ")
  training_size = int(0.1*len(featuresets))
  test_set = featuresets[:training_size]
  training_set = featuresets[training_size:]
  classifier = nltk.NaiveBayesClassifier.train(training_set)
  print ("Accuracy of classifier : ")
  print (nltk.classify.accuracy(classifier, test_set))
  print ("---------------------------------------------------")
  print ("Showing most informative features")
  classifier.show_most_informative_features()
  print_confusionmatrix(classifier,test_set)
  print ("")

def print_confusionmatrix(classifier_type, test_set):
  reflist = []
  testlist = []
  for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier_type.classify(features))
  
  print (" ")
  print ("The confusion matrix")
  cm = ConfusionMatrix(reflist, testlist)
  print (cm)

def create_test_submission(featuresets,test_featuresets,fileName):
  print ("---------------------------------------------------")
  print ("Training and testing a classifier ")
  test_set = test_featuresets
  training_set = featuresets
  classifier = nltk.NaiveBayesClassifier.train(training_set)
  fw = open(fileName,"w")
  fw.write("PhraseId"+','+"Sentiment"+'\n')
  for test,id in test_featuresets:
    fw.write(str(id)+','+str(classifier.classify(test))+'\n')
  fw.close()


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")



def combined_document_features(document, word_features, SL, bigram_features):
  document_words = set(document)
  document_bigrams = nltk.bigrams(document)
  features = {}
  #print(bigram_features[0])

  for word in document_words:
        # features object
    posword = 0
    neutword = 0
    negword = 0
    for word in document_words:
      if word in SL[0]:
        posword += 1
      if word in SL[1]:
        neutword += 1
      if word in SL[2]:
        negword += 1
      features['positivecount'] = posword
      features['neutralcount'] = neutword
      features['negativecount'] = negword

    for word in word_features:
      features['V_{}'.format(word)] = False
      features['V_NOT{}'.format(word)] = False

    for bigram in bigram_features:
      features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)

    return features



if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])

