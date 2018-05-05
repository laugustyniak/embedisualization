from gensim import corpora

preprocess = [strip_proppers(doc) for doc in texts]

tokenized_text = [tokenize(text) for text in preprocess]

texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

print(len(texts[0]))

dictionary = corpora.Dictionary(texts)

dictionary.filter_extremes(no_below=1, no_above=0.8)

corpus = [dictionary.doc2bow(text) for text in texts]

len(corpus)

lda = models.LdaModel(corpus, num_topics=4, id2word=dictionary, update_every=5, chunksize=10000, passes=100)

print(lda[corpus[0]])

topics = lda.print_topics(5, num_words=20)

topics_matrix = lda.show_topics(formatted=False, num_words=20)
