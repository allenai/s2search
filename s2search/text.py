import re
import numpy as np
import pandas as pd
from nltk.util import ngrams
from blingfire import text_to_words
from unidecode import unidecode


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
  }
REGEX_TRANSLATION_TABLE = str.maketrans('', '', "^$.\+*?{}[]()|")


def extract_from_between_quotations(text):
    """Get everything that's in double quotes
    """
    results = re.findall('"([^"]*)"', text)
    return [i.strip() for i in results]


def remove_single_non_alphanumerics(text):
    """Removes any single characters that are not
    alphanumerics and not important punctuation.
    """
    text = re.sub(r"\B[^\w\"\s]\B", "", text)
    return standardize_whitespace_length(text)


def replace_special_whitespace_chars(text: str) -> str:
    """It's annoying to deal with nonbreaking whitespace chars like u'xa0'
    or other whitespace chars.  Let's replace all of them with the standard
    char before doing any other processing."""
    text = re.sub(r"\s", " ", text)
    return text


def standardize_whitespace_length(text: str) -> str:
    """Tokenization is problematic when there are extra-long whitespaces.
    Make them all a single character in length.
    Also remove any whitespaces at beginning/end of a string"""
    return re.sub(r" +", " ", text).strip()


def fix_text(s):
    """General purpose text fixing using nlpre package
    and then tokenizing with blingfire
    """
    if pd.isnull(s):
        return ''
    s = unidecode(s)
    # fix cases when quotes are repeated
    s = re.sub('"+', '"', s)
    # dashes make quote matching difficult
    s = re.sub('-', ' ', s)
    s = replace_special_whitespace_chars(s)
    # tokenize
    s = text_to_words(s).lower().strip()
    # note: removing single non-alphanumerics
    # means that we will match ngrams that are
    # usually separate by e.g. commas in the text
    # this will improve # of matches but also
    # surface false positives
    return remove_single_non_alphanumerics(s)


def fix_author_text(s):
    """Author text gets special treatment.
    No de-dashing, no tokenization, and 
    replace periods by white space.
    """
    if pd.isnull(s):
        return ''
    s = unidecode(s) 
    # fix cases when quotes are repeated
    s = re.sub('"+', '"', s)
    # no periods as those make author first letter matching hard
    s = re.sub(r'\.', ' ', s)
    s = replace_special_whitespace_chars(s)
    s = standardize_whitespace_length(s)
    return text_to_words(s).lower().strip()


def find_query_ngrams_in_text(q, t, quotes=False, len_filter=1, remove_stopwords=True, use_word_boundaries=True, max_ngram_len=7):
    """A function to find instances of ngrams of query q
    inside text t. Finds all possible ngrams and returns their
    character-level span.
    
    Note: because of the greedy match this function can miss
    some matches when there's repetition in the query, but
    this is likely rare enough that we can ignore it
    
    Arguments:
        q {str} -- query
        t {str} -- text to find the query within
        qutes {bool} -- whether to find exact quotes or not
        len_filter {int} -- shortest allowable matches in characters
        remove_stopwords {bool} -- whether to remove stopwords from matches
        use_word_boundaries {bool} -- whether to care about word boundaries
                                      when finding matches
        max_ngram_len {int} -- longest allowable derived word n-grams
    
    Returns:
        match_spans -- a list of span tuples
        match_text_tokenized -- a list of matched tokens
        longest_starting_ngram -- the longest matching ngram that
                                  matches at the start of the text 
    """
    longest_starting_ngram = ''
    
    if len(q) == 0 or len(t) == 0:
        return [], [], longest_starting_ngram
    if type(q[0]) is not str or type(t) is not str:
        return [], [], longest_starting_ngram

    q = [standardize_whitespace_length(i.translate(REGEX_TRANSLATION_TABLE)) 
         for i in q]
    q = [i for i in q if len(i) > 0]
    t = standardize_whitespace_length(t.translate(REGEX_TRANSLATION_TABLE))

    # if not between quotes, we get all ngrams
    if quotes is False:
        match_spans = []
        match_text_tokenized = []
        for q_sub in q:
            q_split = q_sub.split() 
            n_grams = [] 
            longest_ngram = np.minimum(max_ngram_len, len(q_split))
            for i in range(int(longest_ngram), 0, -1): 
                n_grams += [' '.join(ngram).replace('|', '\|')for ngram in ngrams(q_split, i)]
            for i in n_grams:
                if t.startswith(i) and len(i) > len(longest_starting_ngram):
                    longest_starting_ngram = i 
            if use_word_boundaries:
                matches = list(re.finditer('|'.join(['\\b' + i + '\\b' for i in n_grams]), t))
            else:
                matches = list(re.finditer('|'.join(n_grams), t))
            match_spans.extend([i.span() for i in matches
                               if i.span()[1] - i.span()[0] > len_filter])
            match_text_tokenized.extend([i.group()
                                         for i in matches
                                         if i.span()[1] - i.span()[0] > len_filter])
            
        # now we remove any of the results if the entire matched ngram is just a stopword
        if remove_stopwords:
            match_spans = [span for i, span in enumerate(match_spans) if match_text_tokenized[i] not in STOPWORDS]
            match_text_tokenized = [text for text in match_text_tokenized if text not in STOPWORDS]  
        
    # now matches for the between-quotes texts
    # we only care about exact matches
    else:
        match_spans = []
        match_text_tokenized = []
        for q_sub in q:
            if use_word_boundaries:
                matches = list(re.finditer('\\b' + q_sub + '\\b', t))
            else:
                matches = list(re.finditer(q_sub, t))
            if t.startswith(q_sub) and len(q_sub) > len(longest_starting_ngram):
                longest_starting_ngram = q_sub 
            match_spans.extend([i.span() for i in matches])
            match_text_tokenized.extend([i.group() for i in matches]) 
    
    return match_spans, match_text_tokenized, longest_starting_ngram
