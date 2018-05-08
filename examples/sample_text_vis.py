import pandas as pd

from embedisualization.embedisualization import Embedisualisation

if __name__ == '__main__':
    trump_tweets_df = pd.read_csv('realDonaldTrump_tweets.csv')
    emb = Embedisualisation(trump_tweets_df.dropna().text.tolist())
    emb.create_d3_visualisation()
