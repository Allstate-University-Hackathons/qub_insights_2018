# uses python 3

# ---- USGAE ----
# - download tweets data from :  https://www.kaggle.com/thoughtvector/customer-support-on-twitter
#       Make sure you unzip it and put it in the same directory as this python file
#       This download will give you a zipped file containing a sample file and another
#       zipped file. It is this second zipped file we are going to use. Extract it to twcs.csv


# - install this python module https://github.com/carpedm20/emoji


# - assuming you have a complete version of python with pandas, numpy and re, you are ready to run this

#------------------------------------------------------------------------
#                   IMPORTS
#------------------------------------------------------------------------

import numpy as np
import pandas as pd
import emoji
import re

#------------------------------------------------------------------------
    #          SPLITTING FUNCTION
#------------------------------------------------------------------------

def random_train_test_validate_split(df, train_frac=.6, test_frac=.2, seed=1):
    '''
        Function to randomly split the data into [train, test, validate] set, but in reproduceable way
    '''
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_frac * m)
    test_end = int(test_frac * m) + train_end
    train = df.loc[perm[:train_end]]
    test = df.loc[perm[train_end:test_end]]
    validate = df.loc[perm[test_end:]]
    return train, test, validate

#------------------------------------------------------------------------
#                      DATA READ
#------------------------------------------------------------------------

raw = pd.read_csv('twcs.csv')
tweets = raw[['tweet_id','text']]

 # there are 2,811,774 tweets in this database. Probably don't need them all
targetstext = [':smiling_face:',           ':smiling_face_with_heart-eyes:',
               ':face_with_tears_of_joy:', ':face_with_rolling_eyes:',
               ':angry_face:',             ':thinking_face:',
               ':green_heart:',            ':thumbs_down:']
targets = set([emoji.emojize(x) for x in targetstext])

tweets = tweets[tweets.text.apply(lambda x: any(emo in x for emo in targets))]

#------------------------------------------------------------------------
#                       PROCESS
#------------------------------------------------------------------------

tweets.text = tweets.text.apply(lambda x: emoji.demojize(x))
tweets['emoticons']=tweets.text.apply(lambda x:
                                      list(set([':{}:'.format(y) for y in re.findall(r':([^\s]*?):' ,x, re.DOTALL)])))
tweets.text = tweets.text.apply(lambda x: re.sub(r':([^\s]*?):', '', x).replace('  ', ' '))

#------------------------------------------------------------------------
#               FILTER TO SELECTED EMOJIS
#------------------------------------------------------------------------

tweets.emoticons = tweets.emoticons.apply(lambda x: [emo for emo in x if emo in targetstext])

#------------------------------------------------------------------------
#                    SPLIT AND SAVE
#------------------------------------------------------------------------

train, test, validate = random_train_test_validate_split(tweets, train_frac=.8, test_frac=.2, seed=1)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

test_text = test[['tweet_id', 'text']]
test_answers = test[['tweet_id', 'emoticons']]

train.to_csv('train.csv')
test_text.to_csv('test.csv')
test_answers.to_csv('test_answers.csv')