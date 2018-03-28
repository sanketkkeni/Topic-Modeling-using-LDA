# Sanket Keni - Thanks Noah Lidell for providing an excellent explaination
# https://github.com/NoahLidell/math-of-intelligence

import sqlite3
import pandas as pd
from gensim import corpora, models, similarities
import nltk
from collections import Counter

conn = sqlite3.connect('C:\\Users\\Sanket Keni\\Desktop\\Genesis\\math-of-intelligence-master\\generative_models\\nature_articles.db')
cursor = conn.cursor()
num_articles = cursor.execute('SELECT count(distinct title) FROM articles WHERE wc > 1500;').fetchall()[0][0]
print('Number of unquie articles in dataset: ', num_articles)

df = pd.read_sql_query("SELECT distinct(title), text, url, journal, date FROM articles WHERE wc > 1500 ORDER BY random();",conn)
df.head()

subjects = cursor.execute("SELECT distinct topic FROM articles;").fetchall()
print("Subjects in dataset:\n")
for s in subjects:
    print('\t',s[0])


def stem_and_stopword_filter(text, filter_list):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    return [stemmer.stem(word) for word in text.split() if word not in filter_list and len(word) > 2]

def stopword_filter(text, filter_list):
    return [word for word in text.split() if word not in filter_list and len(word) > 2]



def render_topics(subjects, num_topics=3, stem=False, filter_n_most_common_words=500, num_words=30):
    if isinstance(subjects, str):
        df = pd.read_sql_query("SELECT distinct(title), text FROM articles WHERE wc > 1500 and topic = '{}';".format(subjects),
                               conn)
    else:
        df = pd.read_sql_query("SELECT distinct(title), text FROM articles WHERE wc > 1500 and topic IN {};".format(subjects),
                               conn)
    
    docs = df['text'].values
    split_docs = [doc.split(' ') for doc in docs]
    doc_words = [words for doc in split_docs for words in doc] # list of all words in all docs
    wcount = Counter()
    wcount.update(doc_words)
    stopwords = nltk.corpus.stopwords.words('english') + ['introduction','conclusion'] # filter out terms used as section titles in most research papers
    for w, _ in wcount.most_common(filter_n_most_common_words): # remove top 500 words appearing in documents
        stopwords.append(w)
        
    if stem == True:
        docs = [stem_and_stopword_filter(doc, stopwords) for doc in docs]
    else:
        docs = [stopword_filter(doc, stopwords) for doc in docs]
    dictionary = corpora.Dictionary(docs)
    
    corpus = [dictionary.doc2bow(doc) for doc in docs] # doc to bag of words
    # dictionary.doc2idx(["articles","depth"])
    #lda_model = models.LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    
    topics = lda_model.show_topics(formatted=False, num_words=num_words)
    
    print(subjects)
    
    for t in range(len(topics)):
        print("\nTopic {}, top {} words:".format(t+1, num_words))
        print(" ".join([w[0] for w in topics[t][1]]))
        

#specific subjects to analyze for topics as a tuple of strings
# ie subjects = ('philosophy', 'nanoscience-and-technology', 'biotechnology')
subjects = ('philosophy')

render_topics(subjects, num_topics=9, stem=False, filter_n_most_common_words=500)


# topic modeling from group of topics
render_topics(('mathematics-and-computing','computational-biology-and-bioinformatics','nanoscience-and-technology'),
               num_topics=9, stem=False, filter_n_most_common_words=500)



subjects = cursor.execute('SELECT distinct topic FROM articles;').fetchall()

for s in subjects:
    render_topics(s[0], num_topics=9, stem=False, filter_n_most_common_words=500) 
    print('==================================================================================================')























