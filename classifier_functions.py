import tweepy as tw
import re 
import json
import lime
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


# def get_keys(path):
#     with open(path) as f:
#         return json.load(f)

# keys = get_keys('api_keys.json')
# consumer_key = keys['tw_consumer_key']
# consumer_secret = keys['tw_consumer_secret']
# access_token = keys['tw_access_token']
# access_secret = keys['tw_access_secret']
# bearer_token = keys['bearer_token']
# username = keys['s3_username']
# access_key = keys['s3_access_key']
# secret = keys['s3_secret_key']

# auth = tw.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_secret)
# api = tw.API(auth)

def get_tweet_text(api, username):
    tweets = api.user_timeline(screen_name=username, 
                           count=20,
                           include_rts = False,
                           tweet_mode = 'extended')
    text_list = []
    for tweet in tweets:
        text_list.append(tweet.full_text)

    text = ' '.join(text_list)
    return text

def clean_text_for_explaining(text):
    text = re.sub('&amp;', 'and', text)
    text = re.sub('_', ' ', text)
    remove_list = ["(?i)you'll", '(?i)you', '\sI\s', "I'm", "I'll", "I've", '(?i)\swe\s', "(?i)you're", "(?i)\syour\s", "(?i)they'll"
                   "(?i)they\s", "(?i)\stheir", "(?i)there's", "(?i)that's", '@', '(?i)\shtt[^\s]+', '(?i)\swww[^\s]+', '\s[^aI]\s',
                   "(?i)\sthey're", "aren't"] 
    for item in remove_list:
        text = re.sub(r''+item+'', ' ', text)   
    return text

def classify_text(to_classify, model, vectorizer, class_names):
    predictor_pipeline = make_pipeline(vectorizer, model) 
    # class_names = ['analytic', 'continental', 'phenomenology', 'german_idealism', 'plato', 'aristotle', 'empiricism', 'rationalism']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(to_classify, 
                                    predictor_pipeline.predict_proba, 
                                    num_features=8, 
                                    labels=[0, 1, 2, 3, 4, 5, 6, 7],
                                    top_labels=3)
    # exp.show_in_notebook(text=True)
    return exp







