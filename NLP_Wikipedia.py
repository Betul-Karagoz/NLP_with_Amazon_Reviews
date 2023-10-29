
#################################################
# NLP - Text Preprocessing & Text Visualization
#################################################

###################f##############################
# Description of the Problem
#################################################
# Performing text pre-processing, cleaning operations and visualization studies from Wikipedia sample data.


#################################################
# Data Set Story
#################################################
# Contains text taken from Wikipedia data.


############################
# Required Libraries and settings
#########################

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

#Read Data

df = pd.read_csv("DataScience/datasets/wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.shape
df.head()

###############################
# Preprocessing operations on the text
###############################

def clean_text(text):
    # Normalizing Case Folding
    text=text.str.lower()
    # Punctuations
    text=text.str.replace(r'[^\w\s]', '')
    text=text.str.replace("\n", '')
    # Numbers
    text=text.str.replace('\d', '')
    return text

df["text"] = clean_text(df["text"])

df.head()

#######################
#Remove unuseful words from the text
#######################

import nltk
nltk.download('stopwords')
def remove_stopwords(text):

    sw = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    return text

###################
#Find the least repetitive words in the text.
####################

temp_df = pd.Series(' '.join(df['text']).split()).value_counts()

#################
#Remove the least repetitive words from the text.
#################

drops = temp_df[temp_df <= 1000]

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

####################
#Tokenization
####################

nltk.download("punkt")

df["text"].apply(lambda x: TextBlob(x).words).head()

####################
#Lemmatization
####################

nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################
#Visualization
##################

##################
# Barplot
#################

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

tf[tf["tf"] > 5000].plot.bar(x="words", y="tf")
plt.show()

###################
# Wordcloud
###################

text = " ".join(i for i in df.text)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

################

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#######################
#All Processes as a Single Function
#######################

def wikipedia_nlp_prep(text, Barplot = True, Wordcloud=True):

    """
    Performs pre-processing operations on texts.

    :param text: Variable with text in DataFrame
    :param Barplot: Barplot visualization
    :param Wordcloud: Wordcloud visualization
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """

    text = text.str.lower()
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace("\n", '')
    text = text.str.replace('\d', '')
    sw = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    temp_df = pd.Series(' '.join(df['text']).split()).value_counts()
    drops = temp_df[temp_df <= 1000]
    df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

    df["text"].apply(lambda x: TextBlob(x).words).head()
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    tf.sort_values("tf", ascending=False)

    tf[tf["tf"] > 5000].plot.bar(x="words", y="tf")
    plt.show()

    text = " ".join(i for i in df.text)

    wordcloud = WordCloud().generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    wordcloud = WordCloud(max_font_size=50,
                          max_words=100,
                          background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    return text

wikipedia_nlp_prep(df["text"], Barplot=True, Wordcloud=True)
