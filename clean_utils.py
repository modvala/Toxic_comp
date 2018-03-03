import pandas as pd
import numpy as np
import string
import re 
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer


eng_stopwords = set(stopwords.words("english"))

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()

def feat_engine(data):

    data['count_sent']=data["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
    #Word count in each comment:
    data['count_word']=data["comment_text"].apply(lambda x: len(str(x).split()))
    #Unique word count
    data['count_unique_word']=data["comment_text"].apply(lambda x: len(set(str(x).split())))
    #Letter count
    data['count_letters']=data["comment_text"].apply(lambda x: len(str(x)))
    #punctuation count
    data["count_punctuations"] =data["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    #upper case words count
    data["count_words_upper"] = data["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    #title case words count
    data["count_words_title"] = data["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    #Number of stopwords
    data["count_stopwords"] = data["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    #Average length of the words
    data["mean_word_len"] = data["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    #derived features
    #Word count percent in each comment:
    data['word_unique_percent']=data['count_unique_word']*100/data['count_word']
    #derived features
    #Punct percent in each comment:
    data['punct_percent']=data['count_punctuations']*100/data['count_word']
    
    #Leaky features
    data['ip']=data["comment_text"].apply(lambda x: re.findall("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}",str(x)))
    #count of ip addresses
    data['count_ip']=data["ip"].apply(lambda x: len(x))

    #links
    data['link']=data["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
    #count of links
    data['count_links']=data["link"].apply(lambda x: len(x))

    #article ids
    data['article_id']=data["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))
    data['article_id_flag']=data.article_id.apply(lambda x: len(x))

    #username
    ##              regex for     Match anything with [[User: ---------- ]]
    # regexp = re.compile("\[\[User:(.*)\|")
    data['username']=data["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))
    #count of username mentions
    data['count_usernames']=data["username"].apply(lambda x: len(x))
    #check if features are created
    #data.username[data.count_usernames>0]
    
    return data



def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    APPO = {
        "aren't" : "are not",
        "can't" : "cannot",
        "couldn't" : "could not",
        "didn't" : "did not",
        "doesn't" : "does not",
        "don't" : "do not",
        "hadn't" : "had not",
        "hasn't" : "has not",
        "haven't" : "have not",
        "he'd" : "he would",
        "he'll" : "he will",
        "he's" : "he is",
        "i'd" : "I would",
        "i'd" : "I had",
        "i'll" : "I will",
        "i'm" : "I am",
        "isn't" : "is not",
        "it's" : "it is",
        "it'll":"it will",
        "i've" : "I have",
        "let's" : "let us",
        "mightn't" : "might not",
        "mustn't" : "must not",
        "shan't" : "shall not",
        "she'd" : "she would",
        "she'll" : "she will",
        "she's" : "she is",
        "shouldn't" : "should not",
        "that's" : "that is",
        "there's" : "there is",
        "they'd" : "they would",
        "they'll" : "they will",
        "they're" : "they are",
        "they've" : "they have",
        "we'd" : "we would",
        "we're" : "we are",
        "weren't" : "were not",
        "we've" : "we have",
        "what'll" : "what will",
        "what're" : "what are",
        "what's" : "what is",
        "what've" : "what have",
        "where's" : "where is",
        "who'd" : "who would",
        "who'll" : "who will",
        "who're" : "who are",
        "who's" : "who is",
        "who've" : "who have",
        "won't" : "will not",
        "wouldn't" : "would not",
        "you'd" : "you would",
        "you'll" : "you will",
        "you're" : "you are",
        "you've" : "you have",
        "'re": " are",
        "wasn't": "was not",
        "we'll":" will",
        "didn't": "did not",
        "tryin'":"trying"
}
    
    comment=comment.lower()
    comment=re.sub("\\n","",comment)
    #comment = myDate.sub(' xxDATExx ', comment)
    #comment=re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",comment)
    #comment=re.sub("\[\[.*\]","",comment)
    words=tokenizer.tokenize(comment)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    #words = [w for w in words if not w in eng_stopwords]   
    clean_sent=" ".join(words)
    
    return(clean_sent)