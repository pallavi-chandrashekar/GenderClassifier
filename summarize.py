import pickle
import json


def main():

	#get number of tweets collected in collect.py
	Tweets= json.loads(open('tweets.txt').read())
	NumOfTweets = len(Tweets)
	print('Number of Tweets collected = '+str(NumOfTweets))
	
	#Get number of clusters formed in cluster.py
	data= open('Cluster.txt','rb')
	cluster = pickle.load(data)
	NumOfClusters= len(cluster)
	print('Number of Tweets collected = ',NumOfClusters)
	






if __name__ == '__main__':
    main()

