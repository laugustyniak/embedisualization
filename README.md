# Visualize embed of documents

## Install requirements and spacy model

You could use `conda`

`conda create --name embedisualization python=3.6`

`source activate embedisualization`

`pip install -r requirements.txt`

`python -m spacy download en_core_web_sm`

## Install embedisualization lib

`pip install embedisualization`

## Example

To run exemplary visualisation go to `examples` directory and run

`python sample_text_vis.py`

It will take minute or two to generate embeddings and create 2D vis. The new webpage with D3 visualisation will be presented.

![Sample of Trump's Tweets Embedisialized](https://raw.githubusercontent.com/laugustyniak/embedisualization/master/examples/trump.gif)
