import pandas as pd
import ast


"Tried to use pandas to read in data but seperator of +++$+++ wasn't working..."
data = pd.DataFrame()

genre = {}
words = set()
category = []
id = []

"Create list of nltk stop words to remove from text as these words aren't meaningful in this case"
with open('extras.txt', 'r') as f:
    lines = f.readlines()
    for row in lines:
        temp = row.strip('\n')
        words.add(temp)
f.close()


"""Creates dictionary of moveid: genre"""
category_dict = {}
with open('movie_titles_metadata.txt', 'r') as f:
    lines = f.readlines()
    for row in lines:
        movie = row.strip('\n').split(" +++$+++ ")
        id = movie[0]
        category = ast.literal_eval(movie[5])
        category_dict[id]  = category
f.close()

"""Creates dictionary of lineid to text of that line"""
text_dict = {}
with open('movie_lines.txt', 'r') as f:
    lines = f.readlines()
    for row in lines:
        movie = row.strip('\n').split(" +++$+++ ")
        line_id = movie[0]
        text= str.lower(movie[4]).split()
        text = ' '.join([w for w in text if not w in words])
        text_dict[line_id] = text
f.close()

#print(text_dict)

"""Loops through conversations, loops through lines in converstions, counts the number of words in a conversation,
adds it in a dictionary of genre:[conversation length (int)] for each genre a movie takes on"""

with open('movie_conversations.txt', 'r') as f:
    lines = f.readlines()
    for row in lines:
        movie = row.strip('\n').split(" +++$+++ ")
        movieid = movie[2]
        conversations = ast.literal_eval(movie[3])
        count = 0
        for line in conversations:
            string = text_dict[line]
            length = len(string.split())
            count+=length
        for category in category_dict[movieid]:
            if category not in genre:
                genre[category] = [count]
            else:
                genre[category].append(count)
f.close()

"""Average length of all conversations in genre, finds maximum average length """
average = 0
for i in genre:
    genre[i] = sum(genre[i])/len(genre[i])
print(genre)
maximum = max(genre, key=genre.get)
print(maximum)

"""Biography has longest conversations in this case"""


"Extra credit"


from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora
import gensim.models.ldamodel


tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()
text_list = []
"""Sorts all lines and takes the longest 5 lines (top 5)"""
meep = sorted(text_dict, key=lambda k: len(text_dict[k]), reverse=True)
x = meep[:5]

"""Creates list of lists with tokens in each sentence after processing punctuation, stopwords, and uses porter stemming
to decompose words into their roots"""
for i in x:
    text= text_dict[i]
    tokens = tokenizer.tokenize(text)
    remove_stop = [i for i in tokens if not i in words]
    texts = [p_stemmer.stem(i) for i in remove_stop]
    text_list.append(texts)

corpora_dict = corpora.Dictionary(text_list)
# convert tokenized documents into a document matrix
corpus = [corpora_dict.doc2bow(text) for text in text_list]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=corpora_dict, passes=50)
print(ldamodel.print_topics(num_topics=3, num_words=3))
for i in ldamodel.print_topics():
    for j in i:
        print(j)

"""Did not see need to use pandas, however I do use pandas when doing ML work as it makes it easy to index and
separate training and test data."""


