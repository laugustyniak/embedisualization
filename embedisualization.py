from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import mpld3
import pandas as pd
import spacy
from more_itertools import flatten
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from top_toolbar import TopToolbar

CLUSTER_COLORS = {
    0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a',
    4: '#66a61e', 5: '#960000', 6: '#96fffa',
    # 7: '#190000', 8: '#9632fa', 9: '#963214'
}

CSS = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
    """

nlp = spacy.load('en_core_web_sm')


def tokenize_text(text):
    return [
        token.lemma_
        for token
        in nlp(text)
        if not token.is_punct and
           not token.is_digit and
           not token.is_space and
           not token.like_num
    ]


class Embedisualisation:

    def __init__(self, texts: List[str], text_labels: List[str] = None, num_clusters: int = 6):
        self.texts = texts
        self.text_labels = text_labels
        self.num_clusters = num_clusters

        self.dist = None

        if text_labels is None:
            self.text_labels = ['' for _ in self.texts]

    def create_d3_visualisation(self):

        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words=None,
            use_idf=True,
            tokenizer=tokenize_text,
            ngram_range=(1, 3))

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.texts)
        # terms = tfidf_vectorizer.get_feature_names()
        self.dist = 1 - cosine_similarity(tfidf_matrix)

        km = KMeans(n_clusters=self.num_clusters)
        km.fit(tfidf_matrix)
        clusters = km.labels_.tolist()

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(self.dist)  # shape (n_components, n_samples)

        xs, ys = pos[:, 0], pos[:, 1]

        # set up cluster names using a dict
        cluster_names = {i: f'Cluster {i}' for i in range(len(CLUSTER_COLORS))}
        text_label_and_text = [': '.join(t) for t in list(zip(self.text_labels, self.texts))]

        # create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x=xs, y=ys, clusters=clusters, title=text_label_and_text))

        # group by cluster
        groups_clusters = df.groupby('clusters')

        # set up plot
        fig, ax = plt.subplots(figsize=(25, 15))  # set size
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return
        # the appropriate color/label
        for name, group in groups_clusters:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name],
                    color=CLUSTER_COLORS[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(axis='x',  # changes apply to the x-axis
                           which='both',  # both major and minor ticks are affected
                           bottom='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelbottom='off')
            ax.tick_params(axis='y',  # changes apply to the y-axis
                           which='both',  # both major and minor ticks are affected
                           left='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelleft='off')

        ax.legend(numpoints=1)  # show legend with only 1 point

        # add label in x,y position with the label as the
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

        # uncomment the below to save the plot if need be
        plt.savefig(f'clusters_static-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.png', dpi=200)

        # Plot
        fig, ax = plt.subplots(figsize=(28, 20))  # set plot size
        ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        for name, group in groups_clusters:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none',
                             color=CLUSTER_COLORS[name])
            ax.set_aspect('auto')
            labels = [i for i in group.title]

            # set tooltip using points, labels and the already defined 'css'
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=CSS)
            # connect tooltip to fig
            mpld3.plugins.connect(fig, tooltip, TopToolbar())

            # set tick marks as blank
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            # set axis as blank
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        ax.legend(numpoints=1)  # show legend with only one dot
        mpld3.show()  # show the plot

        html = mpld3.fig_to_html(fig)

    def draw_dendogram(self):
        linkage_matrix = ward(self.dist)  # define the linkage_matrix using ward clustering pre-computed distances
        fig, ax = plt.subplots(figsize=(15, 20))  # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=self.text_labels)

        plt.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom='off',  # ticks along the bottom edge are off
                        top='off',  # ticks along the top edge are off
                        labelbottom='off')
        plt.tight_layout()  # show plot with tight layout

        # uncomment below to save figure
        plt.savefig('dendogram_clusters.png', dpi=200)  # save figure as ward_clusters
        plt.close()


if __name__ == '__main__':
    emb = Embedisualisation([
        'This is example. This is subfirst.', 'this is next example', 'this is third example',
        'This is task. This is subfirst.', 'this is next tesk', 'this is third tesk',
    ])
    emb.create_d3_visualisation()
