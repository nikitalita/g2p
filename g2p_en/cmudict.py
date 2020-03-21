import nltk

try:
    nltk.data.find("corpora/cmudict.zip")
except LookupError:
    nltk.download("cmudict")

_cmudict = nltk.corpus.cmudict.dict()


def query_cmudict(word):
    if word in _cmudict:
        return _cmudict[word][0]
    else:
        return None
