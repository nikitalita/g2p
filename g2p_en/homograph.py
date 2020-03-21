from pathlib import Path


homograph_file = Path(__file__).parent / "homographs.en"

homograph_dict = dict()
with homograph_file.open() as f:
    for line in f:
        if line.startswith("#"):
            continue
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph_dict[headword.lower()] = pron1.split(), pron2.split(), pos1


def query_homograph(word, pos):
    if word not in homograph_dict:
        return None
    pron1, pron2, pos1 = homograph_dict[word]
    if pos.startswith(pos1):
        return pron1
    else:
        return pron2
