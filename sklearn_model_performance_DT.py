import sys
import pandas
import numpy
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
def process(filepath):
  # number of folds for cross-validation
  kFolds = 10

  # read in the file with the pandas package
  train_set = pandas.read_csv(filepath)
  
  # this is a data frame for the data
  print ('Shape of feature data - num instances with num features + class label')
  print (train_set.shape)
  
  # convert to a numpy array for sklearn
  train_array = train_set.values
  
  # get the last column with the class labels into a vector y
  train_y = train_array[:,-1]
  
  # get the remaining rows and columns into the feature matrix X
  train_X = train_array[:,:-1]

  print ('** Results from Decision Tree Classifier')

  classifier = DecisionTreeClassifier(criterion = 'gini', max_depth = 7, min_samples_split= 5)
  
  y_pred = cross_val_predict(classifier, train_X, train_y, cv=kFolds)


  print(classification_report(train_y, y_pred))
  
  # confusion matrix from same
  cm = confusion_matrix(train_y, y_pred)
  #print_cm(cm, labels)
  print('\n')
  print(pandas.crosstab(train_y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))

# use a main so can get feature file as a command line argument
if __name__ == '__main__':
    # Make a list of command line arguments, omitting the [0] element
    # which is the script itself.
    args = sys.argv[1:]
    if not args:
        print ('usage: python run_sklearn_model_performance_Decision_Tree.py ')
        sys.exit(1)
    trainfile = args[0]
    process(trainfile)
