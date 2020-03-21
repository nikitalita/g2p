import re
import unicodedata

import nltk

from .expand import normalize_numbers
from .pronunciation import get_pronunciation
from .utils import chain_with_separator

try:
    nltk.data.find("taggers/averaged_perceptron_tagger.zip")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

tokenizer = nltk.tokenize.TweetTokenizer()


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
    tokens = nltk.pos_tag(words)
    phonemes = [get_pronunciation(*x) for x in tokens]
    phonemes = list(chain_with_separator(phonemes, " "))
    return phonemes
