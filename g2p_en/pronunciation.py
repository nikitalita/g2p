import os
from .g2p_core import G2p
from nltk.corpus import cmudict

g2p_core = G2p()
cmudict_ = cmudict.dict()


def construct_homograph_dictionary(filename):
    homograph2features = dict()
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
            headword, pron1, pron2, pos1 = line.strip().split("|")
            homograph2features[headword.lower()] = pron1.split(), pron2.split(), pos1
    return homograph2features


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "homographs.en")
homograph2features = construct_homograph_dictionary(filename)


def check_non_word(word):
    if not word.islower():
        return [word]
    else:
        return None


def query_homograph(word, pos):
    if word not in homograph2features:
        return None
    pron1, pron2, pos1 = homograph2features[word]
    if pos.startswith(pos1):
        return pron1
    else:
        return pron2


def query_cmudict(word):
    if word in cmudict_:
        return cmudict_[word][0]
    else:
        return None


def get_pronunciation(word, pos):
    return (
        check_non_word(word)
        or query_homograph(word, pos)
        or query_cmudict(word)
        or g2p_core.predict(word)
    )
