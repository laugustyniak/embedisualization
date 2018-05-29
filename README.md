# Visualize embed of documents

## Install requirements and spacy model

You could use `conda`

`conda create --name embedisualization python=3.6`

`source activate embedisualization`

`pip install -r requirements.txt`

`python -m spacy download en_vectors_web_lg`

This spacy model [en_vectors_web_lg](https://spacy.io/models/en#en_vectors_web_lg) is quite heavy (631MB). The vocabulary consist of 1.1m keys, 1.1m unique vectors (300 dimensions).

## Install embedisualization lib

`pip install embedisualization`

## Example

To run exemplary visualisation go to `examples` directory and run

`python sample_text_vis.py`

It will take minute or two to generate embeddings and create 2D vis. The new webpage with D3 visualisation will be presented.

![Sample of Trump's Tweets Embedisualized](https://raw.githubusercontent.com/laugustyniak/embedisualization/master/examples/trump.gif)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Flaugustyniak%2Fembedisualization.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Flaugustyniak%2Fembedisualization?ref=badge_shield)


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Flaugustyniak%2Fembedisualization.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Flaugustyniak%2Fembedisualization?ref=badge_large)