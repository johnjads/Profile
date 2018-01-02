from textblob import TextBlob
import re
########################################Tweets##############################
#Prepare your data and follow by

def process_tweets(tweets):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweets).split())

def analyzing_sentiment(tweets):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(process_tweets(tweets))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

    # We create a column with the result of the analysis:
    data['NEW'] = np.array([analyzing_sentiment(tweets) for tweets in data['Tweets']])

    # We display the updated dataframe with the new column:
    print(display(data.head()))