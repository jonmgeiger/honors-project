#Use this module as follows:
#from helpers import path_pdf,path_pkl,find_in_list
#from helpers import plot_freq_dist,get_top_n_words,plot_words_freq
#from helpers import printh,get_best_match,find_start,find_next,
#from helpers import elbow_plot, gridsearch_plot, plot_single_alpha
#from helpers import aggregate_topics,cos_sim,gmm_show_topic

import numpy as np
import re
import os
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


## Utility to find the index of a single word in a list
find_in_list = lambda l, e: l.index(e) if e in l else -1

def plot_freq_dist(freq_list, **kwargs):
    '''
    Plot distribution of word frequencies
    '''
    fig = plt.subplots(figsize=(15,5))
    _ = plt.hist(freq_list, bins=100, **kwargs);
    _ = plt.title("Distribution of word frequencies ("+str(len(freq_list))+" words)");
    _ = plt.xlabel("Word frequency in corpus", {'fontsize': 14});
    _ = plt.ylabel("Log count", {'fontsize': 14});
    plt.yscale('log', nonpositive='clip');
    plt.show();
    return fig

def get_top_n_words(corpus, n_top_words=None):
    '''
    Plot frequency distribution of top n word 
    corpus: list of tokens
    n_top_words: number of most frequent words to plot
    '''
    count_vectorizer = CountVectorizer(stop_words='english')
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx], idx) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return zip(*words_freq[:n_top_words])

def plot_words_freq(word_list, freq_list, n_top_words=20, ylim=None, plot_doc_fraction=False,data_words_bigrams=None,title=True):
    '''
    Plot the frequency of n_top_words in the corpus data, see figure 4 in Odden, Marin, Caballero 2020. 

    Parameters
    ----------
    :param list(str) word_list: list of words in the corpus
    :param list(int) freq_list: frequency of list of words in the corpus
    :param int n_top_words: number of top words to plot
    :param tuple(float) ylim: the limit on the y-axis
    :param bool plot_doc_fraction: Plot the Document fraction on the right axis, default is False
    :param list(list(str)) data_words_bigrams: bigrams for each document, needed if plot_doc_fraction=True
    :return: Tuple (fig, ax) with matplotlib handles to figure and axis
    :rtype: : Tuple[bytes, bytes]
    '''
    fig, ax = plt.subplots(figsize=(8,5))
    word_len = str(len(word_list))
    freq_list = freq_list[:n_top_words]
    word_list = word_list[:n_top_words]
    ax.plot(range(len(freq_list)), freq_list, label='Number of occurrences');
    ax.set_xticks(range(len(word_list)));
    xticks = list(map(lambda w: str(w), word_list));
    ax.set_xticklabels(xticks, rotation=45, ha='right', fontdict={'fontweight': 'normal'});
    if title == True:
        ax.set_title('Top words in corpus (' + word_len + ' total words)', {'fontsize': 16, 'fontweight': 'bold'});
    ax.set_xlabel('Top words', {'fontsize': 14});
    ax.set_ylabel('Number of occurrences (log scale)', {'fontsize': 14});
    ax.set_yscale('log', nonpositive ='clip', );
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([10**(np.floor(np.log10(min(freq_list[:20]))*10)/10), 10**(np.ceil(np.log10(max(freq_list))*10)/10)]);    
    #Fraction
    if plot_doc_fraction:
        frac = [sum([w in temp for temp in data_words_bigrams])/len(data_words_bigrams) for w in word_list[:n_top_words]]
        ax2 = ax.twinx()
        ax2.plot(range(len(freq_list)), frac, 'r', label='Document fraction');
        ax2.set_ylim([np.floor(min(frac)*10)/10,1])
        ax2.set_ylabel('Document fraction', {'fontsize': 14}, labelpad=10);
        plt.legend(loc='upper left')    
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')    
    plt.tight_layout()
    plt.show();
    return fig, ax


## Fuzzy string match
from difflib import SequenceMatcher
def get_best_match(query, corpus, step=4, flex=3, case_sensitive=False, verbose=False):
    """Return best matching substring of corpus.
    Credits to the accepted answer here: https://stackoverflow.com/questions/36013295/find-best-substring-match

    Parameters
    ----------
    :param str query:
    :param str corpus:
    :param int step: Step size of first match-value scan through corpus. Can be thought of
        as a sort of "scan resolution". Should not exceed length of query.
    :param int flex: Max. left/right substring position adjustment value. Should not
        exceed length of query / 2.
    :return output0: Best matching substring.
    :rtype: str
    :return output1: Match ratio of best matching substring. 1 is perfect match.
    :rtype: float
    """
    def _match(a, b):
        """Compact alias for SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def scan_corpus(step):
        """Return list of match values from corpus-wide scan."""
        match_values = []

        m = 0
        while m + qlen - step <= len(corpus):
            match_values.append(_match(query, corpus[m : m-1+qlen]))
            if verbose:
                print(query, "-", corpus[m: m + qlen], _match(query, corpus[m: m + qlen]))
            m += step

        return match_values

    def index_max(v):
        """Return index of max value."""
        return max(range(len(v)), key=v.__getitem__)
    
    def index_maxima(v, n=5):
        """Return indices of n max values."""
        #change index_max() to return a list of indices for the n highest values of the input list, 
        #and loop over adjust_left_right_positions() for values in that list.
        max_ind = np.argpartition(v, -n)[-n:] #unsorted indices of n maxima (n=4)
        
        max_ind = list(np.argpartition(v, -n)[-n:]) #unsorted indices of n maxima (n=4)
        return [max_ind[i] for i in np.argsort(np.array(np.array(v))[max_ind])]
        
    
    def adjust_left_right_positions():
        """Return left/right positions for best string match."""
        # bp_* is synonym for 'Best Position Left/Right' and are adjusted 
        # to optimize bmv_*
        #bp_ls, bp_rs, matches = [] ,[], []
        #for pos in positions:
        p_l, bp_l = [pos] * 2
        p_r, bp_r = [pos + qlen] * 2

        # bmv_* are declared here in case they are untouched in optimization
        r = int(p_l / step)
        if int(r) != r: print(ratio is not integer.investigate)
        bmv_l = match_values[r]
        bmv_r = match_values[r]

        for f in range(flex):
            ll = _match(query, corpus[p_l - f: p_r])
            if ll > bmv_l:
                bmv_l = ll
                bp_l = p_l - f

            lr = _match(query, corpus[p_l + f: p_r])
            if lr > bmv_l:
                bmv_l = lr
                bp_l = p_l + f

            rl = _match(query, corpus[p_l: p_r - f])
            if rl > bmv_r:
                bmv_r = rl
                bp_r = p_r - f

            rr = _match(query, corpus[p_l: p_r + f])
            if rr > bmv_r:
                bmv_r = rr
                bp_r = p_r + f

            if verbose:
                print("\n" + str(f))
                print("ll: -- value: %f -- snippet: %s" % (ll, corpus[p_l - f: p_r]))
                print("lr: -- value: %f -- snippet: %s" % (lr, corpus[p_l + f: p_r]))
                print("rl: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r - f]))
                print("rr: -- value: %f -- snippet: %s" % (rl, corpus[p_l: p_r + f]))
            #bp_ls.append(bp_l)
            #bp_rs.append(bp_r)
            #matches.append(_match(query, corpus[bp_l : bp_r]))
        return bp_l, bp_r, _match(query, corpus[bp_l : bp_r])
        #return bp_ls, bp_rs, matches
        
    if not case_sensitive:
        query = query.lower()
        corpus = corpus.lower()

    qlen = len(query)

    if flex >= qlen/2:
        print("Warning: flex %d exceeds length of query / 2 = %d. Setting to default. query=%s" % 
              (flex, qlen/2, query))
        flex = 3

    match_values = scan_corpus(step)
    pos = index_max(match_values) * step
    #positions = list(map(lambda x: x * step, index_maxima(match_values)))
    pos_left, pos_right, match_value = adjust_left_right_positions()
    return (pos_left,pos_right), corpus[pos_left: pos_right].strip(), match_value


#Print with highlighted regex
def printh(text, pattern='', crop = -1, print_pos = False):
    '''Print text, highlight a pattern, crop around all found patterns.
    
    :param str text:
    :param str pattern: regex pattern
    :param str crop: int for trimming text around patterns found, default is -1
    :param str print_pos: boolean for printing positions in text of found patterns, default is False
    '''
    ms = [m for m in re.finditer(pattern, text)]
    highlight_start = '\x1b[1;31;43m'
    highlight_end = '\x1b[0m'
    if print_pos:
        print([m.start() for m in ms])
    if len(ms) is 0:
        print('\033[1mprinth: pattern \''+pattern+'\' not found\x1b[0m')
        return
    if crop > -1:
        for m in ms:
            cropped_start = max(0, m.start() - crop)
            cropped_end = min(len(text), m.end() + crop)
            temp = text[cropped_start:m.start()] + highlight_start + text[m.start():m.end()] + highlight_end + text[m.end():cropped_end]
            print(temp,'\n')
    else :
        for m in reversed(ms):
            text = text[:m.start()] + highlight_start+ text[m.start():m.end()] + highlight_end + text[m.end():]    
        print(text,'\n')

def find_title(text, title, verbose=False, threshold=0.67):
    """
    Find a `title` string (article's title) in a `text` string (article's text). If no match is found, perform Fuzzy substring match and return the best (NB: not the first!) match with score > threshold. 
    
    :param str text: String text
    :param str title: Substring to find in string
    :param bool verbose: Default is False
    :param float threshold: Threshold score for accepting fuzzy substring match, default is 0.67
    :return startloc: Start and end positions of title match. -1 if no match    
    :rtype: list[int]
    """
    vprint = lambda msg: print(msg) if verbose else None
    # Search in text
    titleloc = (-1,-1)
    #Use regex or fuzzy to find next title
    ms = [m for m in re.finditer(title, text)]
    if len(ms)>1:
        vprint('%d matches found on the next title. Choose the last. %s' % 
            (len(ms), [(m.start(), m.end()) for m in ms]))
        ms=[ms[-1]]
    if len(ms) == 1: 
        titleloc=(ms[0].start(), ms[0].end())
        vprint('    - Title found at %d-%d' % 
                          (titleloc[0], titleloc[1]))
    else:
        #Use Fuzzy substring match
        # NB: because author2find can be short, applying get_best_match would usually show a warning
        vprint('     - Fuzzy substring search for title')
        vprint('     - title: %s' % re.sub(r'[\t\n\r\f\v\uf0b7\\]', '', title))
        bestmatch = get_best_match(title, text)
        if bestmatch[2] > threshold:
            titleloc = bestmatch[0]
            vprint('     - position fuzzy match=%s' % str(titleloc))
        else:
            verbose = True
        vprint('     - Low matching'*(bestmatch[2] <= threshold) + '  scor %ef' % bestmatch[2])
        vprint('     - fuzzy: %s' % re.sub(r'[\t\n\r\f\v\uf0b7\\]', '', bestmatch[1]))
    return titleloc

def find_start(DF, index, text_col='raw', title='title', authors='authors', verbose=False, apply_lower=True, threshold=0.67, dist_auth_title=1E6, chars_overlap=-1):
    """
    Find title or author for the document in a DataFrame <DF> at <index>,<text_col> by finding an exact match 
    and return the first match. If no exact match is found, perform Fuzzy substring match and return the best 
    (NB: not the first!) match. 

    :param DataFrame DF: pandas DataFrame containing the columns: title, authors
    :param int index: Index of DF. Cannot be the first index using the .loc property
    :param str text_col: Column name containing text, default is 'raw'
    :param str title: Title or column name containing the title
    :param str authors: Authors or column name containing author to search in text, default 'authors'
    :param bool verbose: Default is False
    :param bool apply_lower: Convert text and patterns to lowercase, default is True
    :param float threshold: Threshold score for accepting fuzzy substring match, default is 0.67
    :param int dist_auth_title: Maximum number of chars allowed between authors and title found in text, default 1E6
    :param int chars_overlap: Authors and title are searched in text overlap between previous and current document. DF.loc[index-1, text_col][0:chars_overlap], default is -1, i.e. use the whole text
    :return startloc: Position of text match. -1 if no match
    :rtype: int
    """
    vprint = lambda msg: print(msg) if verbose else None
    # Get text to search into. Try to find text overlap between previous and current article
    text = DF.loc[index, text_col]    
    pos_end_prev = DF.loc[index-1, text_col].find(DF.loc[index, text_col][0:chars_overlap])
    if pos_end_prev > 0:
        text = text[0:len(DF.loc[index-1, text_col][max(pos_end_prev,0):])]
        vprint('%d - Text overlap between previous and current article is [0:%d]. The current article is %d long' % 
            (index, len(text), len(DF.loc[index, text_col])))
    # Get title and author to find
    title2find = title
    if title in DF.columns:
        title2find = F.loc[index+1][title]
    title2find = re.escape(title2find)
    author2find = authors
    if authors in DF.columns:
        author2find = DF.loc[index]['authors'].split(' ')[0]
    author2find += '\W'
    # Find authors. Use regex
    startloc, authloc, titleloc = -1, (-1,-1), (-1,-1)
    ms = [m for m in re.finditer(author2find, text)]
    if len(ms) == 0:
        # Sometimes there are symbols after an author's name
        author2find = author2find[0:-2]
        ms = [m for m in re.finditer(author2find, text)]
    vprint('%d - author2find=%s' % (index, author2find))
    if len(ms)==0:
        vprint('%d - Quit. 0 matches found on authors: %s' % (index, author2find))
        return startloc
    elif len(ms)>1:
        vprint('%d - %d matches found on authors: %s. Take the first one. %s' % 
            (index, len(ms), author2find, [(m.start(), m.end()) for m in ms]))
        ms = [ms[0]]
    authloc=(ms[0].start(), ms[0].end())
    vprint('%d - Authors found at %d-%d. index=%d' % 
        (index, authloc[0], authloc[1], index))

    # Use regex or fuzzy find to find title 
    if apply_lower: 
        text, title2find = [el.lower() for el in [text, title2find]]
    titleloc = find_title(text=text, title=title2find, verbose=verbose, threshold=threshold)

    # Check that title and authors locations are close.
    if authloc[0] > -1 and titleloc[0] > -1:
        dist = abs(min(titleloc[1]-authloc[0], authloc[1]-titleloc[0]))
        vprint('%d - Distance between title and authors found is %d' % (index, dist))
        if not dist_auth_title or dist < dist_auth_title:
            startloc = min(authloc[0], titleloc[0])
            vprint('%d - Successfully detected start location: %d - %s' % 
                (index, startloc, DF.loc[index, 'filename']))
        else:
            vprint('%d - .. which cannot be accepted because higher than dist_auth_title=%d' % (index, dist_auth_title))  
    return startloc


def find_next(DF, index, text_col='raw', title='title', authors='authors', verbose=False, apply_lower=True, threshold=0.67, dist_auth_title=1E6): 
    #chars_overlap=0):
    """
    Find the following article's title or author for the document at <index>,<text_col> in a DataFrame <DF> by right-finding an exact match and (NB!) return the first match. 
    If no match is found, perform Fuzzy substring match and return the best (NB: not the first!) match with score > threshold. 
    
    :param DataFrame DF: DataFrame containing the columns: title, authors
    :param int index: Index of DF
    :param str text_col: Column name containing text
    :param str title: Title or column name containing the title
    :param str authors: Authors or column name containing author to search in text, default 'authors'
    :param bool verbose: Default is False
    :param bool apply_lower: Convert text and patterns to lowercase, default is True
    :param float threshold: Threshold score for accepting fuzzy substring match, default is 0.67
    :param int dist_auth_title: Maximum number of chars allowed between authors and title found in text, default 1E6
    #:param int chars_overlap: Authors and title are searched in text overlap between current and next document. DF.loc[index, text_col][-chars_overlap:], default is 0, i.e. use the whole text
    :return startloc: Position of text match. -1 if no match    
    :rtype: int
    """
    vprint = lambda msg: print(msg) if verbose else None
    # Get text to search into. Try to find text overlap between current and next article
    text = DF.loc[index, text_col]#[-chars_overlap:] #1 page is typically <=4000 chars
    # Get title and author to find
    title2find = title
    if title in DF.columns:
        title2find = DF.loc[index+1][title]
    title2find = re.escape(title2find)
    author2find = authors
    if authors in DF.columns:
        author2find = DF.loc[index+1]['authors'].split(' ')[0]
    author2find += '\W'
    # Find authors. Use regex
    endloc, authloc, titleloc = -1, (-1,-1), (-1,-1)
    ms = [m for m in re.finditer(author2find, text)]
    if len(ms) == 0:
        # Sometimes there are symbols after an author's name
        author2find = author2find[0:-2]
        ms = [m for m in re.finditer(author2find, text)]
    vprint('%d - author2find=%s' % (index, author2find))
    if len(ms)==0: 
        vprint('%d - Quit. 0 matches found on authors: %s' % (index, author2find))
        return endloc
    elif len(ms)>1: 
        vprint('%d - %d matches found on the next author. Choose the last. %s' % 
            (index, len(ms), 
            [(m.start(), m.end()) for m in ms]))
            #[(max(0, len(DF.loc[index, text_col])-chars_overlap)+m.start(), 
            #    max(0, len(DF.loc[index, text_col])-chars_overlap)+ m.end()) for m in ms]))
        ms = [ms[-1]]
    authloc=(ms[0].start(),ms[0].end())
    vprint('%d - Authors found at %d-%d' % (index, authloc[0], authloc[1]))
    
    #Use regex or fuzzy to find next title
    if apply_lower: 
        text, title2find = [el.lower() for el in [text, title2find]]
    titleloc = find_title(text, title2find, verbose=verbose, threshold=threshold)

    # Check that title and authors locations are close.
    if authloc[0] > -1 and titleloc[0] > -1:
        dist = abs(min(titleloc[1]-authloc[0], authloc[1]-titleloc[0]))
        vprint('%d - Distance between title and authors found is %d' % (index,dist))
        if not dist_auth_title or dist < dist_auth_title:
            endloc = min(authloc[0], titleloc[0])
            vprint('%d - Successfully detected end location: %d - %s' % 
                (index, endloc, DF.loc[index, 'filename']))
        else:
            vprint('%d - .. which cannot be accepted because higher than dist_auth_title=%d' % (index, dist_auth_title))  
    return endloc #+ max(0, len(DF.loc[index, text_col])-chars_overlap)
    #print("NB: filter_start and filter_end return the best fuzzy match, which is not necessarily the first or the last match.")


def elbow_plot(df0, ylim=None):
        ks = df0.num_topics.unique().tolist()
        reps = len(df0[(df0.num_topics == df0.iloc[0].num_topics)])
        #Print scatter plot
        for i,k in enumerate(ks):
            label=(i==0) and 'Coherence' or '_nolabel_'
            plt.scatter([k]*reps, df0[df0.num_topics == k].coherence, c="black", label=label);
        plt.errorbar(ks, df0.groupby('num_topics').coherence.mean(), yerr=df0.groupby('num_topics').coherence.std(), fmt='--o', label="Mean");
        plt.ylabel("Coherence score");
        plt.xlabel("num_topics");
        plt.legend(loc="best")
        plt.ylim(ylim);
        
def gridsearch_plot(df, no_below = None, aggreg_func = 'mean'):
    '''
    Plot the results of a grid search. For each no_below value and each alpha value, plot all coherences by topic number
    aggreg_func - 'mean' or 'median'. Aggregation function for the coherence values on the y-axis
    '''    
    nbs = [None]
    alphas = [None]
    if no_below: 
        df = df[df.no_below == no_below]
    if 'no_below' in df.columns:
        reps = int(len(df)/len(df.no_above.unique())/len(df.no_below.unique())/len(df.num_topics.unique()))
        #reps = len(df[(df.num_topics == df.iloc[0].num_topics) & (df.no_below == df.iloc[0].no_below)])
        nbs = df.no_below.unique()
    else: 
        reps = len(df[(df.num_topics == df.iloc[0].num_topics)])
    if 'alpha' in df.columns:
        reps = int(reps / len(df.alpha.unique()))
        alphas = df.alpha.unique()        
    for nb in nbs:
        from matplotlib.ticker import MaxNLocator
        if nb: df0 = df[df.no_below == nb]
        for alpha in alphas:
            if alpha: df0 = df[df.alpha == alpha]
            x = df0.num_topics.unique()
            y = df0.coherence
            fig, ax1 = plt.subplots()
            for i in range(reps):
                _ = ax1.scatter(x, y[i::reps], c="black");
            if aggreg_func == 'mean':
            	coh_mean = y.groupby(np.arange(len(y))//reps).mean()
            elif aggreg_func == 'median':
            	coh_mean = y.groupby(np.arange(len(y))//reps).median()
            print('Maximum %f at num_topics=%d ' % (max(coh_mean), x[0]+list(coh_mean).index(max(coh_mean))))
            _ = ax1.plot(x, coh_mean, c="blue", label="mean");
            _ = ax1.errorbar(x, coh_mean, yerr=y.groupby(np.arange(len(y))//reps).std(), fmt='--o')
            _ = ax1.set_ylabel("Coherence score");
            _ = ax1.set_xlabel("num_topics");
            _ = ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            _ = plt.legend(loc="best")
            _ = plt.title("Coherence score. no_below=%s" % (str(nb)))
            if alpha: _ = plt.title("Coherence score. no_below=%s, alpha=%s" % (str(nb), str(alpha)))
            plt.show();

def plot_single_alpha(df0, ax, alpha):
    df0 = df0[df0.alpha == alpha]
    ks = df0.num_topics.unique().tolist()
    reps = len(df0[(df0.num_topics == df0.iloc[0].num_topics)])
    #Print scatter plot for each alpha
    for i,k in enumerate(ks):        
        label=(i==0) and 'Coherence' or '_nolabel_'
        ax.scatter([k]*reps, df0[df0.num_topics == k].coherence, s=60, facecolors='none', edgecolors='k', label=label);
    ax.errorbar(ks, df0.groupby('num_topics').coherence.mean(), yerr=df0.groupby('num_topics').coherence.std(), fmt='--o', label="Mean");
    _ = ax.set_ylabel("Coherence score", {'fontsize': 14});
    _ = ax.set_xlabel("Number of topics", {'fontsize': 14});
    _ = ax.legend(loc="best")
    _ = ax.set_title(r"$\alpha=%s$" % (str(alpha)), {'fontsize': 14})
    _ = plt.ylim([0.39, 0.53])


def aggregate_topics(matrix_topics, lbls, n_components = None):
    '''Aggregate (i.e. average) topics based on labels. For example, pass a cluster model's labels and this 
     function will return a (K', n_words) matrix where each column is the average of all points a clusters'''
    if not n_components:
        n_components = max(lbls)+1
    else:
        if n_components != max(lbls)+1:
            raise ValueError("%d labels inconsistent with n_components (%d)" % (max(lbls)+1,n_components))
    return [np.mean(matrix_topics[lbls == l], axis=0) for l in range(n_components)]

from numpy.linalg import norm #import linear algebra norm
cos_sim = lambda v1,v2: np.inner(v1, v2) / (norm(v1) * norm(v2)) #define cosine similarity

def gmm_show_topic(model_cluster, topicid, hypertopics, topn = 20):
    """Adapting LdaModel.show_topic to a clustering model"""
    idx = np.argsort(hypertopics[topicid])[::-1][:topn] #Find the cluster centers for each cluster, and locate the ids for their words
    values = hypertopics[topicid][idx] 
    topic = [(str(id2word[i])+'*'+'{:5.3}'.format(hypertopics[topicid][i])) for i in idx]  #build the topic list from those ids      
    topic = [(id2word[i],hypertopics[topicid][i]) for i in idx]  #build the topic list from those ids      
    return topic