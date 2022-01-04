import requests
from semanticdist import utils
from urllib.error import URLError
import numpy as np
import pnadas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


def get_single_entities(text, context):
    _params = {
        'text': text,
        'lang': 'AGNOSTIC',
        'annType': 'NAMED_ENTITIES',
        'key': context['babelfy_key']
    }
    resp = requests.post(
        "https://babelfy.io/v1/disambiguate", data=_params).json()
    if "message" in resp:
        return None
    return resp


def get_entities(data, part, context, pickle_file=None):
    """Get semantic entities for a text field.

    We can only get some entities per day on API, so just get as many as we can
    """
    if f'{part}_entities' in data:
        needed_indexes = data.index[(~data[part].isnull()) & (
            data[f'{part}_entities'].isnull())]
    else:
        needed_indexes = data.index[~data[part].isnull()]
    found_indexes = []
    entities = []
    for i in needed_indexes:
        try:
            print(f"Looking for index {i}")
            e = get_single_entities(
                data.loc[i, part], context)
        except URLError:
            print("Semantic entity request failed with urlerror.")
            break
        if e is None:
            print("Semantic entity request failed.")
            break
        print(f"Found entity for index {i}: {e}")
        found_indexes = found_indexes + [i]
        entities = entities + [e]
    data.loc[found_indexes, f'{part}_entities'] = np.array(
        entities, dtype=object)
    if pickle_file is not None:
        utils.save_data(data, pickle_file, context)
    return data


def augment_entities(entities, text):
    for e in entities:
        e['text_fragment'] = text[e.get('charFragment').get(
            'start') - 5:e.get('charFragment').get('end')+6]


def get_similarity_matrix(entities):
    dfs = [pd.DataFrame(d) for d in entities]
    mlb = MultiLabelBinarizer()
    label_vectors = mlb.fit_transform(df['babelSynsetID'] for df in dfs)
    return cosine_similarity(label_vectors)