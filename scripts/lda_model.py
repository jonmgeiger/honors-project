# Print out all expressions
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Wider Cells
from IPython.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
# Ignore some warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning);

# General Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle as pkl
import copy

# NLP Packages
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel

import nltk
from nltk.corpus import wordnet
from nltk import pos_tag

from scripts.helper_functions import plot_freq_dist,get_top_n_words,plot_words_freq

def text_to_bigrams(df): 
    def ML_process(text): 
        filt_text = text
        # Remove quotes, commas, periods, parentheses
        filt_text = re.sub('[",\.\(\)/]', '', filt_text)
        # Remove single quotes
        filt_text = re.sub("'", '', filt_text)
        return filt_text
    
    def sent_to_words(list_sentences):
        return [gensim.utils.simple_preprocess(str(sentence), deacc=True) for sentence in list_sentences] 

    def remove_stopwords(tokens):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in STOPWORDS ] for doc in tokens]

    def get_wordnet_pos(word): #Provide a POS tag
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN) #return NOUN by default

    def lemmatize_token(token):
        return nltk.stem.WordNetLemmatizer().lemmatize(token, get_wordnet_pos(token))

    def lemmatize(token_list):
        '''Input example: ["he", "matches", "the", "profile"]'''
        return [lemmatize_token(token) for token in token_list]
    
    def make_bigrams(list_tokenized_docs): 
        return [bigram[doc] for doc in list_tokenized_docs]

    df['replaced'] = df.resp.map(lambda x: ML_process(x))
    data_words = sent_to_words(df['replaced'])
    data_words_nostops = remove_stopwords(data_words)
    data_words_lemmatized = [lemmatize(token_list) for token_list in data_words_nostops] 

    phrases = gensim.models.phrases.Phrases(data_words_lemmatized, min_count = 10)
    bigram = gensim.models.phrases.Phraser(phrases)
    data_words_bigrams = make_bigrams(data_words_lemmatized)

    return data_words_bigrams



def bigrams_to_corpus(bigrams):
    corpus = [ w for doc in bigrams for w in doc ]
    return corpus



def plot_words(corpus): 
    # Display the total number of words and unique words
    print('Total words: \033[1m%d\033[0m, unique words: \033[1m%d\033[0m' % (len(corpus), len(set(corpus))))
    # Plot the frequency distributions and the most common words
    [words, freq, ids] = get_top_n_words(corpus, n_top_words=None)
    fig = plot_freq_dist(freq)
    fig, ax = plot_words_freq(words, freq)



def display_words_removed_at_thresholds(data_words_bigrams): 
    no_above_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    no_below_values = [i for i in range(2, 10)]

    print("\nNO ABOVE THRESHOLDS\n\n")

    # NO ABOVE filtering object
    id2word_unfiltered_above = gensim.corpora.Dictionary(data_words_bigrams) #Needed for a bug
    id2word_unfiltered_above.filter_extremes(no_below=0, no_above=1, keep_n=100000)
    
    print("Dictionary with \033[1mno_above=1\033[0m contains \033[1m%d\033[0m unique words\n" % (len(id2word_unfiltered_above)))

    # Initial Filtering
    no_above = 1.0
    id2word = gensim.corpora.Dictionary(data_words_bigrams)
    id2word.filter_extremes(no_below=0, no_above=no_above, keep_n=100000)
    diff_set = set(list(id2word_unfiltered_above.values())) - set(list(id2word.values()))
    print("Dictionary filtered with \033[1mno_above=%.2f\033[0m contains \033[1m%d\033[0m unique words. \033[1m%d words\033[0m removed:\n %s \n" % 
        (no_above, len(id2word), len(diff_set), sorted(diff_set)))


    # Other Values of no_above
    for val in no_above_values: 
        no_below = 0
        no_above = val
        id2word_prev = copy.deepcopy(id2word)
        id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000) #NB: no problem filtering again on no_above
        diff_set = set(list(id2word_prev.values())) - set(list(id2word.values()))
        print("Dictionary filtered with \033[1mno_above=%.2f\033[0m contains \033[1m%d\033[0m unique words. \033[1m%d words\033[0m removed:\n %s \n" % 
            (no_above, len(id2word), len(diff_set), sorted(diff_set)))
    
    print("\n\nNO BELOW THRESHOLDS\n\n")

    # NO BELOW filtering object
    id2word_unfiltered_below = gensim.corpora.Dictionary(data_words_bigrams) #Needed for a bug
    id2word_unfiltered_below.filter_extremes(no_below=1, no_above=no_above, keep_n=100000)
    print("Dictionary with \033[1mno_below=1\033[0m contains \033[1m%d\033[0m unique words\n" % (len(id2word_unfiltered_below)))



    # Initial filtering 
    no_below = 1
    id2word = gensim.corpora.Dictionary(data_words_bigrams)
    id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000)
    diff_set = set(list(id2word_unfiltered_below.values())) - set(list(id2word.values()))
    print("Dictionary filtered with \033[1mno_below=%.2f\033[0m contains \033[1m%d\033[0m unique words. \033[1m%d words\033[0m removed:\n %s \n" % 
        (no_below, len(id2word), len(diff_set), sorted(diff_set)))


    # Other values of no_below

    for val in no_below_values: 
        no_above = 1
        no_below = val
        id2word_prev = copy.deepcopy(id2word)
        id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000) #NB: no problem filtering again on no_above
        diff_set = set(list(id2word_prev.values())) - set(list(id2word.values()))
        print("Dictionary filtered with \033[1mno_below=%.2f\033[0m contains \033[1m%d\033[0m unique words. \033[1m%d words\033[0m removed:\n %s \n" % 
            (no_below, len(id2word), len(diff_set), sorted(diff_set)))



def create_bow(corpus, bigrams, no_above = 0.5, no_below = 3, display_removed_words = False): 

    id2word = gensim.corpora.Dictionary(bigrams)
    id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=100000)
    bow_corpus = [id2word.doc2bow(doc) for doc in bigrams]

    if(display_removed_words == True): 
        diff_set = set(list(id2word_unfiltered.values())) - set(list(id2word.values()))
        print("\nApplying \033[1mno_above={} and no_below={} removes {} words\033[0m:\n {} \n".format(no_above, no_below, len(diff_set), sorted(diff_set)))
        print("Filtered Dictionary contains \033[1m{}\033[0m unique words\n".format(len(id2word)))

    return bow_corpus, id2word



def LDA_model(bow_corpus, id2word, bigrams, num_topics): 
    lda_model = LdaModel(corpus = bow_corpus, 
                         id2word = id2word, 
                         num_topics = num_topics, 
                         alpha = 'auto', 
                         passes = 100, 
                         random_state = 12)
    coherence = CoherenceModel(model = lda_model, 
                               texts = bigrams, 
                               dictionary = id2word, 
                               coherence = 'c_v').get_coherence()
    print("Coherence of model with {} topics:".format(num_topics), coherence)
    print("\n")
    return lda_model



def display_n_responses(df, num_responses_displayed, lda_model, corpus, bow): 
    def get_weight_in_doc(ldamodel, corpus, doc_num, fill=0):
        """ Get the weigth for each topic in a document
        Inputs: ldamodel, corpus, doc_num, fill (value when topic was not found)
        Output: list of tuples [(<topic number>, <topic weight>)]
        """
        num_topics = ldamodel.num_topics
        weights = [fill] * num_topics
        # Get a list of tuples with the detected topics and weights
        row = ldamodel[corpus][doc_num]
        # Process the output to return a list of weights
        tw = list(zip(*row))
        for topic in range(num_topics):
            if topic in tw[0]:
                ind = tw[0].index(topic)
                weights[topic] = tw[1][ind]
        return weights

    def get_weight_per_doc(ldamodel, corpus):
        """ Get a document by topic weight Dataframe 
        Inputs: ldamodel, corpus
        """
        num_topics = ldamodel.num_topics
        dataframe = pd.DataFrame(np.zeros((len(corpus),num_topics)), columns=list(range(num_topics)))
        dataframe.index.name = 'Topic weights'
        for i, row in dataframe.iterrows():
            dataframe.iloc[i] = pd.Series(get_weight_in_doc(ldamodel, corpus, i))
        return dataframe

    weights = get_weight_per_doc(lda_model, bow)
    df_weighted = pd.concat([df, weights], axis = 1)

    def pretty_print(df):
        return display(HTML(df.to_html().replace("\\n","<br>")))

    for t in range(lda_model.num_topics): 
        print('Topic {}: {}'.format(t, lda_model.print_topics(num_words=10)[t][1]))
        pretty_print(df_weighted.sort_values(by=t, ascending=False).loc[:, df_weighted.columns!='replaced'][:num_responses_displayed])
        print('\n\n')  