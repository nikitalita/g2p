import re
import unicodedata

from nltk import pos_tag
from nltk.tokenize import TweetTokenizer

from .expand import normalize_numbers
from .g2p_core import G2p

tokenizer = TweetTokenizer()
g2p_core = G2p()


def strip_accents(text):
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def preprocess_text(text):
    text = str(text)
    text = normalize_numbers(text)
    text = strip_accents(text)
    text = text.lower()
    text = re.sub("[^ a-z'.,?!\-]", "", text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    return text


def g2p(text):
    text = preprocess_text(text)
    words = tokenizer.tokenize(text)
    tokens = pos_tag(words)
    return g2p_core(tokens)
