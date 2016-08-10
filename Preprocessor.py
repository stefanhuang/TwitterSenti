"""
TwitterPrepocessor, a module contains the tokeniser and feature extractor for Task A and B
"""

import nltk
import re
# from nltk.tokenize import TweetTokenizer
from custom_tokenize import TweetTokenizer # customised nltk tokeniser 
from nltk.corpus import stopwords as stopwordloader
from nltk.stem import WordNetLemmatizer
import string
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import opinion_lexicon

stopwords = stopwordloader.words('english')

# puntuaion we choose to ignore, we keep '?' and '!'
punc = list('"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~') 

# tokenise the given text into a list of tokens
def tokenise(tweet, begin=0, end=0, more_instances=0, lemmatization=False):  
    
    tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
    tokens = tknzr.tokenize(tweet.encode('utf-8'))
    
    if begin!=0 or end!=0:
        slicing = [token for token in tokens if token not in string.punctuation+"..."]
        tokens = slicing[ max(0, begin-more_instances) : end+1+more_instances ]
    
    # remove punctuation
    tokens = filter(lambda word: word not in punc, tokens)

    # replace numbers, link, At
    tokens = [u'LINK' if re.match(r"http.*", x) else x for x in tokens]
    tokens = [u'AT_USER' if re.match(r"@.*", x) else x for x in tokens]
    tokens = [u'NUMBER' if re.match(r"\d", x) else x for x in tokens]
    
    # change words to lowercase except all uppercase words
    tokens = [ x.lower() if not x.isupper() else x for x in tokens ]

    # Lemmatization
    if lemmatization:
        tokens = [WordNetLemmatizer().lemmatize(x) if not x.isupper() else x for x in tokens ]

    return tokens
 
class extract_ngram(BaseEstimator, TransformerMixin):
    """ Custom Transformer to extract N-gram feature extractor """

    def __init__(self, ngram_range=(1,1), hashing=False):
        self.ngram_range = ngram_range
        self.hashing = hashing

    def fit(self, x, y=None):
        return self

    def transform(self, tokenlist):
        featurelist = []
        
        for tokens in tokenlist:
            features = {}  
            tokens = [ word.strip('#') for word in tokens ] #strip hashtag
            
            for n in range(self.ngram_range[0], self.ngram_range[1]+1):
            
                ngrams = zip(*[tokens[i:] for i in range(n)])
                for tup in ngrams:
                    key = ' '.join(tup)
                    if self.hashing:
                        features[key] = True
                    else:
                        features[key] = features.get(key, 0) +1
            featurelist.append(features)

        return featurelist

class extract_kskip_bigram(BaseEstimator, TransformerMixin):
    """ Custom Transformer to calculate k skip bigram """   

    def __init__(self, skip=1, hashing=False):
        self.skip = skip
        self.hashing = hashing

    def fit(self, x, y=None):
        return self

    def transform(self, tokenlist):
        featurelist = []
        
        for tokens in tokenlist:
            features = {}  
            tokens = [ word.strip('#') for word in tokens ] #strip hashtag
            
            for s in range(self.skip+1):
            
                ngrams = zip(*[tokens[i:] for i in (0,1+s)])
                for tup in ngrams:
                    key = ' '.join(tup)
                    if self.hashing:
                        features[key] = True
                    else:
                        features[key] = features.get(key, 0) +1
            featurelist.append(features)

        return featurelist

class lexicon_sentiwordnet(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to calculate lexicon scores from unigram
    look up in nltk.corpus.sentiwordnet and 
    reads the maximum positive, negative, and objective score
    feature='all' returns all pos, neg, and obj score
    feature='score' returns one feature SenScore(w) = PMI(w,pos) - PMI(w,neg)
    """    
    def __init__(self, feature='all'):
        self.feature = feature

    def fit(self, x, y=None):
        return self

    # helper method
    def get_lexi_feature(self, tokens):
        senti_list = lambda word: list(swn.senti_synsets(word)) 
        pos = sum([ max([0.0]+[ x.pos_score() for x in senti_list(word) ]) for word in tokens ])
        neg = sum([ max([0.0]+[ x.neg_score() for x in senti_list(word) ]) for word in tokens ])
        
        if self.feature == 'all':
            obj = sum([ max([0.0]+[ x.obj_score() for x in senti_list(word) ]) for word in tokens ])
            return [ pos, neg, obj ]
        if self.feature == 'pos_neg':
            return [ pos, neg ]
        else:
            return [ pos - neg ]
            
    def transform(self, tokenlist):
        return [ self.get_lexi_feature(tokens) for tokens in tokenlist ]

class lexicon_liuhu(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to count positive and negative lexicons
    in the Hu and Liu opinion lexicon
    """
    def __init__(self):
        # use set to improve performace
        self.pos_lexicon = set(opinion_lexicon.positive())
        self.neg_lexicon = set(opinion_lexicon.negative())
    
    def fit(self, x, y=None):
        return self
            
    def transform(self, tokenlist):
        def get_pos_neg(tokens):
            pos = sum(1 for word in tokens if word in self.pos_lexicon)
            neg = sum(1 for word in tokens if word in self.neg_lexicon)
            return [float(pos), float(neg)]
        
        return [ get_pos_neg(tokens) for tokens in tokenlist ]

class lexicon_emoticon(BaseEstimator, TransformerMixin):
    """Custom Transformer to extract lexicon for emoticon and count emoticons"""
    
    LEXICON_PATH = 'Lexicon_And_WE/EmoticonSentimentLexicon/EmoticonSentimentLexicon.txt'

    EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
      |
      [\*#\O~\-=>][_\.]+[\*#\O~\-=<]      # Japanese style Emoticons eye mouth eye
      |
      [\.][_]+[\.]               # poker face
      |
      \^[_.;]?\^                 # smiling eyes
    )"""
    EMOTICON_RE = re.compile(EMOTICONS, re.VERBOSE | re.I | re.UNICODE) # compile for use

    NOISE = ['p;','::', ':@', 'P8', '/8', 'do:']

    def __init__(self):
        self.lexicon = self.load()

    def load(self):
        with open(self.LEXICON_PATH) as lexifile:
            lexicon = {}
            for aline in lexifile:
                line = aline.strip().split('\t')
                lexicon[line[0]] = int(line[1])  # add to dictionary
            return lexicon

    def fit(self, x, y=None):
        return self
    
    def transform(self, tokenlist):
        def getList(tokens):
            count = 0.0
            score = 0.0
            for word in tokens:
                if self.EMOTICON_RE.match(word) and word not in self.NOISE: 
                    count += 1
                    score += self.lexicon[word] if word in self.lexicon else 0
            return [ count, score ]
        return [ getList(tokens) for tokens in tokenlist ]      

class lexicon_NRC_unigram(BaseEstimator, TransformerMixin):
    """Custom Transformer to extract auto generated lexicon from NRC-Canada team's work
         NRC Hashtag Sentiment Lexicon (version 0.1) and  Sentiment140 Lexicon (version 0.1)
        Lexicon downloaded form http://www.saifmohammad.com/WebPages/ResearchInterests.html
    """
    
    HASHTAG_PATH = 'Lexicon_And_WE/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt'
    SENTI140_PATH = 'Lexicon_And_WE/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt'

    EXCLUDED = re.compile(r"((http.*)|(\d)|(@.*))")

    def __init__(self):
        self.lexicon = {}
        self.load(self.HASHTAG_PATH)
        self.load(self.SENTI140_PATH)

    def load(self, filepath):
        with open(filepath) as lexifile:
            for aline in lexifile:
                line = aline.strip().split('\t')
                key = line[0]
                if not self.EXCLUDED.match(key):
                    self.lexicon[key] = float(line[1])  # add to dictionary

    def fit(self, x, y=None):
        return self
    
    def transform(self, tokenlist):
        def getList(tokens):
            scores = [ self.lexicon[word] for word in tokens if word in self.lexicon ]
            return [ sum(scores), min(scores), max(scores) ] if scores else [0,0,0]
        
        return [ getList(tokens) for tokens in tokenlist ]              

class lexicon_NRC_bigram(BaseEstimator, TransformerMixin):
    """Custom Transformer to extract auto generated lexicon from NRC-Canada team's work
         NRC Hashtag Sentiment Lexicon (version 0.1) and  Sentiment140 Lexicon (version 0.1)
        Lexicon downloaded form http://www.saifmohammad.com/WebPages/ResearchInterests.html
    """
    
    HASHTAG_PATH = 'Lexicon_And_WE/NRC-Hashtag-Sentiment-Lexicon-v0.1/bigrams-pmilexicon.txt'
    SENTI140_PATH = 'Lexicon_And_WE/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt'

    EXCLUDED = re.compile(r"((http.*)|(\d)|(@.*))")

    def __init__(self):
        self.lexicon = {}
        self.load(self.HASHTAG_PATH)
        self.load(self.SENTI140_PATH)

    def load(self, filepath):
        with open(filepath) as lexifile:
            for aline in lexifile:
                line = aline.strip().split('\t')
                key = line[0]
                words = key.split(' ')
                if not self.EXCLUDED.match(words[0]) and not self.EXCLUDED.match(words[1]):
                    self.lexicon[key] = float(line[1])  # add to dictionary

    def fit(self, x, y=None):
        return self
    
    def transform(self, featurelist):
        def getList(features):
            scores = [ self.lexicon[bigram] for bigram in features if bigram in self.lexicon ]
            return [ sum(scores), min(scores), max(scores) ] if scores else [0,0,0]
        
        return [ getList(features) for features in featurelist ]

class WE_GloVe_Twitter(BaseEstimator, TransformerMixin):
    """Custom Transformer to extract pretrained GloVe: Global Vectors for Word Representation
        for twitter from glove.twitter.27B.zip http://nlp.stanford.edu/projects/glove/
        Original data downloaded from glove.twitter.27B.zip
        (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)
        The data used here is the custom cleaned version to remove all non-english words
        The cleaned file used in this class is included and we only use 25 dimension for size and speed
    """
    
    GLOVE_Cleaned = 'Lexicon_And_WE/glove.twitter.27B/glove.twitter.25d-cleaned-en.txt'

    def __init__(self):
        self.we = {}
        self.dimension = 25
        self.load(self.GLOVE_Cleaned)

    def load(self, filepath):
        with open(filepath) as wefile:
            for aline in wefile:
                line = aline.strip().split(' ')
                key = line[0]
                vectors = [ float(line[i]) for i in range(1, self.dimension+1)]
                self.we[key] = vectors  # add to dictionary

    def fit(self, x, y=None):
        return self
    
    def transform(self, tokenlist):
        def getList(tokens):
            # process hashtag
            tokens = [ '<hashtag>' if re.match(r"#.*", token) else token for token in tokens ]
            
            dimensions = zip(*[ self.we[token] for token in tokens if token in self.we ])

            return [ np.mean(d) for d in dimensions ] + \
                [ min(d) for d in dimensions ] + \
                [ max(d) for d in dimensions ] if dimensions else [0]*self.dimension*3
        
        return [ getList(tokens) for tokens in tokenlist ]

class extract_tweeter_related(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to get tweet lengths, hashtag counts and etc
    """
    def fit(self, x, y=None):
        return self
    
    def transform(self, tokenlist):
        def getList(tokens):
            wordcount = len(tokens)
            exclam = question = allupper = 0.0
            link = atuser = number = hashtag = 0.0
            for word in tokens:
                if word=='!': 
                    exclam += 1.0
                elif word=='?': 
                    question += 1.0
                elif word=='LINK': 
                    link += 1.0
                elif word=='AT_USER': 
                    atuser += 1.0
                elif word=='NUMBER': 
                    number += 1.0
                elif word.isupper(): 
                    allupper += 1.0    
                elif re.match(r"#.*", word): 
                    hashtag += 1.0
                
            return [ wordcount,exclam,question,allupper,
                    link,atuser,number,hashtag ]    

        return [ getList(tokens) for tokens in tokenlist ]
