
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import pickle
import requests
from pprint import pprint
import re
from scipy.sparse import lil_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from itertools import product
import json

def get_names():
    males_url = 'http://www2.census.gov/topics/genealogy/' +             '1990surnames/dist.male.first'
    females_url = 'http://www2.census.gov/topics/genealogy/' +               '1990surnames/dist.female.first'

    males = requests.get(males_url).text.split('\n')
    females = requests.get(females_url).text.split('\n')

    male_names = []
    female_names = []
    # Get names.

    males_pct = dict([(m.split()[0].lower(), float(m.split()[1])) for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1])) for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or females_pct[f] > males_pct[f]])



    return male_names, female_names

def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):
    """ Split a tweet into tokens."""
    if not string:
        return []
    if lowercase:
        string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()

def sample_tweets(c, male_names, female_names):
    tweets = []
    for k,v in c.items():
        name = get_first_name(v)
        if name in male_names or name in female_names:
            tweets.append(v)
    return tweets

def gender_by_name(tweets, male_names, female_names):
    Names = defaultdict(list)
    m = []
    f = []
    for v in tweets:
        v['gender'] = 'unknown'
    for v in tweets:
        name = v['user']['name']
        if name:
            # remove punctuation.
            name_parts = re.findall('\w+', name.split()[0].lower())
            if len(name_parts) > 0:
                first = name_parts[0].lower()
                if first in male_names:
                    m.append(first)
                    v['gender'] = 'male'
                elif first in female_names:
                    f.append(first)
                    v['gender'] = 'female'
                else:
                    v['gender'] = 'unknown'

    Names['male'] = m
    Names['female'] = f
    gender = []
    for v in tweets:
        gender.append(v['gender'])
    #gender = [v['gender'] for k, v in tweets.items()]
    counts = Counter(gender)
    print('%.2f of accounts are labeled with gender' %
          ((counts['male'] + counts['female']) / sum(counts.values())))

    print('gender counts:\n', counts)
    count_list = []
    count_list.append(counts['male'])
    count_list.append(counts['female'])
    #data = open('Classify_Statistics.json','wb')
    #pickle.dump(counts,data)
    with open('Classify_Statistics.json', 'w', encoding = 'utf-8') as output:
        json.dump(counts, output)
    #data.close()
    #data = open('Name.txt','wb')
    #pickle.dump(Names,data)
    #data.close()
    with open('Names.json', 'w', encoding = 'utf-8') as output1:
        json.dump(Names, output1)
    

def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
    """ Convert a tweet into a list of tokens, from the tweet text and optionally the
    user description. """
    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def run_all(tweets, y, use_descr, lowercase,
            keep_punctuation, descr_prefix,
            collapse_urls, collapse_mentions):
    tokens_list = [tweet2tokens(v, use_descr, lowercase,
                                keep_punctuation, descr_prefix,
                                collapse_urls, collapse_mentions)
                   for v in tweets]
    vocabulary = make_vocabulary(tokens_list)
    X = make_feature_matrix(tweets,tokens_list, vocabulary)
    acc = do_cross_val(X, y, 5)

    #print('acc=', acc)
    return acc

def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    for train_idx, test_idx in cv:
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary))  # If term not present, assign next int.
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]  # looking up a key; defaultdict takes care of assigning it a value.
    #print('%d unique terms in vocabulary' % len(vocabulary))
    return vocabulary


def make_feature_matrix(tweets, tokens_list, vocabulary):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()

def get_raw_data():
    data= open('Data.txt','rb')
    c = pickle.load(data)
    
    return c


def get_gender(tweet, male_names, female_names):
    name = get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1



def main():
    c = get_raw_data()
    male_names, female_names = get_names()
    tweets = sample_tweets(c, male_names, female_names)
    gender_by_name(tweets, male_names, female_names)


    use_descr_opts = [True, False]
    lowercase_opts = [True, False]
    keep_punctuation_opts = [True, False]
    descr_prefix_opts = ['d=', '']
    url_opts = [True, False]
    mention_opts = [True, False]
    y = np.array([get_gender(t, male_names, female_names) for t in tweets])
    argnames = ['use_descr', 'lower', 'punct', 'prefix', 'url', 'mention']
    option_iter = product(use_descr_opts, lowercase_opts,
                          keep_punctuation_opts,
                          descr_prefix_opts, url_opts,
                          mention_opts)
    results = []
    for options in option_iter:
        option_str = '\t'.join('%s=%s' % (name, opt) for name, opt in zip(argnames, options))

        acc = run_all(tweets, y, *options)
        results.append(acc)

    print("Accuracy = " +str(results[:1]))



if __name__ == '__main__':
    main()

