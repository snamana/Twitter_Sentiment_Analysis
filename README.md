# Natural Language Processing  
**Sentiment Analysis on Movie Reviews**

## Overview  
This project performs sentiment analysis on movie reviews from the Rotten Tomatoes dataset, originally created by Socher et al. and used in a Kaggle competition. The goal is to classify movie review phrases into sentiment categories ranging from negative to positive using multiple NLP feature engineering methods and classification algorithms including Naive Bayes, Logistic Regression, and Decision Trees.

## Table of Contents  
- Introduction  
- Dataset Description  
- Feature Engineering  
- Classification Methods  
- Comparative Analysis  
- Conclusion  

## Introduction  
The dataset contains movie review phrases labeled with five sentiment categories:  
0 - Negative  
1 - Somewhat Negative  
2 - Neutral  
3 - Somewhat Positive  
4 - Positive  

Phrases are parsed using the Stanford parser and the goal is to accurately classify unseen phrases in the test set using trained models.

## Dataset Description  
- **Source**: Kaggle (Sentiment Analysis on Movie Reviews)  
- **Size**: ~156,060 phrases  
- **Format**: TSV files with `PhraseId`, `SentenceId`, `Phrase`, and `Sentiment`  
- **Labels**: Five-level sentiment scale

## Feature Engineering  
The project explored several feature extraction techniques:  
1. **Unigram Features** (With and Without Preprocessing)  
2. **Bigram and Trigram Features**  
3. **Negation Word Features**  
4. **POS Tag Features**  
5. **Sentiment Lexicon Features** (MPQA Subjectivity Lexicon)  
6. **Combined Feature Sets**

Tokenization, filtering, and preprocessing (lowercasing, punctuation removal, stopword removal) were applied using regular expressions and NLTK. Vocabulary lists and subjectivity scores were used to enhance feature vectors.

## Classification Methods  

### Naive Bayes Classifier  
- **Best Accuracy**: 90% using POS features  
- Other accuracies ranged from 60% to 70% for unigram, bigram, and sentiment lexicon features.

### Logistic Regression  
- **Solver**: L-BFGS  
- **Class Weight**: Balanced  
- **Max Iterations**: 1000  
- **Best Performance**: Sentiment Lexicon features with F1-score of 0.55

### Decision Tree Classifier  
- **Criterion**: Gini  
- **Max Depth**: 7  
- **Min Samples Split**: 5  
- **Best Performance**: Preprocessed, Bigram, Trigram features with F1-score of 0.36–0.44

## Comparative Analysis  

| Feature Set Type           | Logistic Regression F1 | Decision Tree F1 |
|----------------------------|------------------------|------------------|
| Normal (no preprocessing) | 0.46                   | 0.43             |
| Preprocessed               | 0.42                   | 0.36             |
| Bigram                     | 0.42                   | 0.36             |
| Negation Words             | 0.53                   | 0.50             |
| Sentiment Lexicon          | 0.53                   | 0.59             |
| Trigram                    | 0.42                   | 0.36             |
| POS Tags                   | 0.42                   | 0.42             |
| Combined Features          | 0.00                   | 0.44             |

## Conclusion  
- The Naive Bayes classifier achieved the highest accuracy of 90% using POS-tagged features.  
- Logistic Regression performed best with sentiment lexicon features.  
- Decision Trees showed moderate performance with preprocessed, bigram, and trigram features.  
- Reducing sentiment classes to three (Negative, Neutral, Positive) may improve accuracy in future work.

## Lessons Learned  
- Gained experience with a variety of NLP feature sets and classification models.  
- Applied Python programming techniques for text classification.  
- Understood how dataset size affects model performance and execution time.  
- Tested and evaluated multiple algorithms to compare feature effectiveness for sentiment analysis.
