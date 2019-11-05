from TwitterAPI import TwitterAPI
import sys
import configparser

def getTwitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    config_file = "twitter.cfg"
    config = configparser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI(config.get('twitter', 'consumer_key'),
                config.get('twitter', 'consumer_secret'),
                config.get('twitter', 'access_token'),
                config.get('twitter', 'access_token_secret'))
    return twitter
    
def getRequest(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

