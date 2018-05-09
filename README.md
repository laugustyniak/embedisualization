# Visualize embed of documents

## Install requirements and spacy model

You could use `conda`

`conda create --name embedisualization python=3.6`

`source activate embedisualization`

`pip install -r requirements.txt`

`python -m spacy download en_vectors_web_lg`

This spacy model [en_vectors_web_lg](https://spacy.io/models/en#en_vectors_web_lg) is quite heavy (631MB)

## Install embedisualization lib

`pip install embedisualization`

## Example

To run exemplary visualisation go to `examples` directory and run

`python sample_text_vis.py`

It will take minute or two to generate embeddings and create 2D vis. The new webpage with D3 visualisation will be presented.

![Sample of Trump's Tweets Embedisualized](https://raw.githubusercontent.com/laugustyniak/embedisualization/master/examples/trump.gif)
