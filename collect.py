import twitterCalls
import json


def getData(twitter):
    """ Collect tweets, call robust_request method 50 times to collect more data
        Args:
            twitter..... A TwitterAPI object.
        This function doesn't return anything, but get tweets and save it in "tweets.json"
    """
    
    twitterRequest = twitterCalls.getRequest(twitter, 'search/tweets', {'q': '#GOT8', 'lang': 'en', 'count': 100})
    tweets = dict()
    number = 0
    print("Starting the collection of data")
    for i in twitterRequest:
        print("....")
        tweets[number] = i
        number += 1
    

    for ran in range(49):
        twitterRequest = twitterCalls.getRequest(twitter, 'search/tweets', {'q': '#GOT8', 'lang': 'en', 'count': 100, 'max_id' : tweets[number-1]['id']})
        print("Collecting data for ", (ran+2) ," time")
        for r in twitterRequest:
            tweets[number] = r
            print("....")
            number += 1
    
    print("Total number of tweets collected", len(tweets))
	#tweets are collected
	
	#save the tweets in tweets.json file
    print("Storing tweets into tweets.txt")
                
    with open('tweets.txt', 'w', encoding = 'utf-8') as output:
        json.dump(tweets,output)
    print("Done saving the tweets")


def main():
    #Twitter = getTwitter()
	Twitter = twitterCalls.getTwitter()
	getData(Twitter)
    
if __name__ == '__main__':
    main()