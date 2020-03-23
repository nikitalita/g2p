from .cmudict import query_cmudict
from .homograph import query_homograph
from .predict import predict


def check_non_word(word):
    if not word.islower():
        return [word]
    else:
        return None


def get_pronunciation(word, pos):
    return (
        check_non_word(word)
        or query_homograph(word, pos)
        or query_cmudict(word)
        or predict(word)
    )
