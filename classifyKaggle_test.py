import os
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from classifyKaggle import create_test_submission, get_word_features, usual_features, pre_processing_documents, get_words_from_test


def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./test.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  testphrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      templist=[]
      if len(line.split('\t'))==3:
          templist.append(line.split('\t')[0])
          templist.append(line.split('\t')[2])
          testphrasedata.append(templist)
      else:
          templist.append(line.split('\t')[0])
          templist.append("")
          testphrasedata.append(templist)
  phraselist=testphrasedata

  print('Read', len(testphrasedata), 'phrases, using', len(phraselist), 'test phrases')
  #for phrase in phraselist[:10]:
    #print (phrase)
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []

  # add all the phrases
  for id,phrase in phraselist:

    # with pre processing
    tokenizer = RegexpTokenizer(r'\w+')
    phrase = pre_processing_documents(phrase)
    tokens = tokenizer.tokenize(phrase)
    phrasedocs.append((id, tokens))

  
  # possibly filter tokens

  word_features = get_word_features(preprocessedTokens)
  featuresets = [(normal_features(d, word_features), s) for (d, s) in phrasedocs]
  
  preprocessedTestTokens = get_words_from_test(phrasedocs)
  test_word_features = get_word_features(preprocessedTestTokens)

  test_featuresets=[(normal_features(d, test_word_features),id) for (id,d) in phrasedocs]
  create_test_submission(featuresets,test_featuresets,"output.csv")

