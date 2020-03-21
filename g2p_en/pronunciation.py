from .cmudict import query_cmudict
from .g2p_core import G2p
from .homograph import query_homograph

g2p_core = G2p()


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
        or g2p_core.predict(word)
    )
