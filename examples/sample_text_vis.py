import pandas as pd

from embedisualization.embedisualization import Embedisualisation

if __name__ == '__main__':
    trump_tweets_df = pd.read_csv('realDonaldTrump_tweets.csv')
    # filter some tweets
    trump_tweets_df = trump_tweets_df[trump_tweets_df.text.str.len() > 30].sample(1000)
    emb = Embedisualisation(trump_tweets_df.text.tolist())
    emb.create_d3_visualisation()
