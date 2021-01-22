import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import os
import plotly.express as px
import pickle
import json
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from keras.preprocessing.sequence import pad_sequences
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

from classifier_functions import *

# set styling
external_stylesheets = [dbc.themes.CERULEAN]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# class TextsToSequences(BaseEstimator, TransformerMixin):
#     def __init__(self,  tokenizer):
#         self.tokenizer = tokenizer
        
#     def fit(self, texts, y=None):
#         return self
    
#     def transform(self, texts, y=None):
#         return np.array(self.tokenizer.texts_to_sequences(texts))


# class Padder(BaseEstimator, TransformerMixin):
#     def __init__(self, maxlen=500):
#         self.maxlen = maxlen
#         self.max_index = None
        
#     def fit(self, X, y=None):
#         self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
#         return self
    
#     def transform(self, X, y=None):
#         X = pad_sequences(X, maxlen=self.maxlen)
#         return X

# pull keys from environment
consumer_key = os.environ['TWITTER_CONSUMER_KEY']
consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
access_token = os.environ['TWITTER_ACCESS_TOKEN']
access_secret = os.environ['TWITTER_ACCESS_SECRET']


# model_path = 'model_data\classification_models\\NN_weights_epoch_09_0.7928.hdf5'
# model = load_model(model_path)

# model = load_model('NN_weights_epoch_09_0.7928.hdf5')

# with open('gru_tokenizer.pkl', 'rb') as f:
#     tokenizer = pickle.load(f)

with open('NB_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)
# with open('random_forest.pkl', 'rb') as f:
#     rf_model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# set up classification explanation pipeline
# padder = Padder(450)
# sequencer = TextsToSequences(tokenizer)
# pipeline = make_pipeline(sequencer, padder, model)

# set up labels
school_label_dict = {'analytic': 0,
 'aristotle': 1,
 'capitalism': 2,
 'communism': 3,
 'continental': 4,
 'empiricism': 5,
 'german_idealism': 6,
 'phenomenology': 7,
 'plato': 8,
 'rationalism': 9}


# Twitter Auth
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tw.API(auth)



# twitter search bar
twitter_bar = html.Div(id="twitter-bar-container", children=
    [dbc.Row([
        dbc.Col(html.H3('Twitter Search'), width=2),
        dbc.Col(dbc.Input(id="twitter-bar", 
                    placeholder="enter a twitter username to search",
                    type='text',
                    n_submit=0,
                    style={'width': '100%'},), width=8),
        dbc.Col(dbc.Button("SUBMIT", id="twitter-bar-submit-button", color="primary", className="mr-1", n_clicks=0), width=2)
    ])
])


# search bar object
search_bar = html.Div(id="classification-bar-container", children=
    [
        dcc.Textarea(id="classification-bar", 
                    placeholder="Enter text to classify - the more you can provide, the more accurate the classification will be. \n\nNote that some very common words will be removed from your submission to improve classification accuracy.",
                    # type='text',
                    # n_submit=0,
                    style={'width': '100%', 'height': 600},),
        dbc.Button("SUBMIT", id="classification-bar-submit-button", color="primary", className="mr-1", n_clicks=0)
    ])




app.layout = html.Div([
    dcc.Tabs(id='classifier-tabs', value='tab-text-input', children=[
        dcc.Tab(label='Enter Your Own Text', value='tab-text-input'),
        dcc.Tab(label='Search for a Twitter User', value='tab-twitter')]),
    html.Div(id='classifier-tabs-content')
])

@app.callback(Output('classifier-tabs-content', 'children'),
              [Input('classifier-tabs', 'value')])
def render_content(tab):
    if tab == 'tab-twitter': return [html.Br(), 
                                    twitter_bar, html.Div(id='twitter-bar-output', children=[])]
    if tab == 'tab-text-input': return [html.Br(), html.H3("Classification by Text"),
                                            dbc.Row([
                                                    dbc.Col(search_bar, width=5),
                                                    dbc.Col(html.Div(id="output", children=[html.Div(id='classification-bar-output', children=[])]))
                                                    ])]     

# callback for twitter search bar
@app.callback(Output(component_id="twitter-bar-output", component_property="children"),
              [Input(component_id="twitter-bar-submit-button", component_property="n_clicks"),
              Input(component_id="twitter-bar", component_property="n_submit")],
              [State(component_id="twitter-bar", component_property="value")])
def generate_explainer_html(n_clicks, n_submit, username, 
                            api=api, model=nb_model, vectorizer=vectorizer,
                            class_names = [name.replace('_', ' ').title() for name in list(school_label_dict.keys())]):
    if n_clicks < 1 and n_submit < 1:
        return [html.Br(), html.P('The classification can take some time. Please be patient, and your text classification will appear here when it is ready.')]
    if n_clicks > 0 or n_submit > 0:
        # try:
            tweets = get_tweet_text(api, username)   
            text = clean_text_for_explaining(tweets)
            # class_names = [name.replace('_', ' ').title() for name in list(school_label_dict.keys())]
            # explainer = lime_text.LimeTextExplainer(class_names=class_names,
            #                                         bow=False,
            #                                         split_expression=r'[\s+|\.\s+|,\s+]')
            # exp = explainer.explain_instance(text, 
            #                                 pipeline.predict, 
            #                                 num_features=10, 
            #                                 labels=[0,1,2,3,4,5,6,7,8,9],
            #                                 top_labels=3)
            exp = classify_text(text, model, vectorizer, class_names=class_names)
            obj = html.Iframe(
                srcDoc=exp.as_html(),
                width='100%',
                height='800px',
                style={'border': '2px #d3d3d3 solid'},
            )
            return obj
        # except:
            # return 'Sorry, something went wrong.'



# callback for search bar
@app.callback(Output(component_id="classification-bar-output", component_property="children"),
              [Input(component_id="classification-bar-submit-button", component_property="n_clicks"),
              Input(component_id="classification-bar", component_property="n_submit")],
              [State(component_id="classification-bar", component_property="value")])
def generate_explainer_html(n_clicks, n_submit, username, 
                            api=api, model=nb_model, vectorizer=vectorizer,
                            class_names = [name.replace('_', ' ').title() for name in list(school_label_dict.keys())]):
    if n_clicks < 1 and n_submit < 1:
        return [html.Br(), html.P('The classification can take some time. Please be patient, and your text classification will appear here when it is ready.')]
    if n_clicks > 0 or n_submit > 0:
        # try:
            tweets = get_tweet_text(api, username)   
            text = clean_text_for_explaining(tweets)
            # class_names = [name.replace('_', ' ').title() for name in list(school_label_dict.keys())]
            # explainer = lime_text.LimeTextExplainer(class_names=class_names,
            #                                         bow=False,
            #                                         split_expression=r'[\s+|\.\s+|,\s+]')
            # exp = explainer.explain_instance(text, 
            #                                 pipeline.predict, 
            #                                 num_features=10, 
            #                                 labels=[0,1,2,3,4,5,6,7,8,9],
            #                                 top_labels=3)
            exp = classify_text(text, model, vectorizer, class_names=class_names)
            obj = html.Iframe(
                srcDoc=exp.as_html(),
                width='100%',
                height='800px',
                style={'border': '2px #d3d3d3 solid'},
            )
            return obj
        # except:
            # return 'Sorry, something went wrong.'



server = app.server

if __name__ == '__main__':
    app.config.suppress_callback_exceptions = True
    app.run_server(debug=True)                                               