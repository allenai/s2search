# s2search
The Semantic Scholar Search Reranker

The code in this repo is for when you have a plain-text query and some academic documents, 
and your goal is to search within the documents and obtain a score for how 
good of a match each document is for the query. The standard pipeline involves a first-stage ranker (like ElasticSearch) and a reranker. 
The model included with this repository is for the reranking stage only, but you may have few-enough documents 
that a first-stage ranker is not necessary. The model and featurization are both fast.

## Installation
To install this package, run the following:

```bash
git clone https://github.com/allenai/s2search.git
cd s2search
conda create -y --name s2search python==3.7
conda activate s2search
python setup.py develop
pip install https://github.com/kpu/kenlm/archive/master.zip
```

To obtain the necessary data, run this command after the package is installed:

`aws s3 cp --no-sign-request s3://ai2-s2-research-public/s2search_data.zip .`

Then unzip the file. Iniside the zip is folder named `s2search/` that will contain all of the artifacts you'll need to get predictions.

Warning: this zip file is 10G compressed and 17G uncompressed.

## Example
Warning: you will need more than 17G of ram because of the large `kenlm` models that need to be loaded into memory.

An example of how to use this repo:

```
from s2search.rank import S2Ranker

# point to the artifacts downloaded from s3
data_dir = 's2search/'

# the data is a list of dictionaries
papers = [
    {
        'title': 'Neural Networks are Great',
        'abstract': 'Neural networks are known to be really great models. You should use them.',
        'venue': 'Deep Learning Notions',
        'authors': ['Sergey Feldman', 'Gottfried W. Leibniz'],
        'year': 2019,
        'n_citations': 100,
        'n_key_citations': 10
    },
    {
        'title': 'Neural Networks are Terrible',
        'abstract': 'Neural networks have only barely worked and we should stop working on them.',
        'venue': 'JMLR',
        'authors': ['Isaac Newton', 'Sergey Feldman'],
        'year': 2009,
        'n_citations': 5000  # we don't have n_key_citations here and that's OK
    }
]

# only do this once because we have to load the giant language models into memory
s2ranker = S2Ranker(data_dir)

# higher scores are better
print(s2ranker.score('neural networks', papers))
print(s2ranker.score('feldman newton', papers))
print(s2ranker.score('jmlr', papers))
print(s2ranker.score('missing', papers))
```

Note that `n_key_citations` is a Semantic Scholar feature. If you don't have it, just leave that key out of the data dictionary. The other paper fields are required.
