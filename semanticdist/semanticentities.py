import requests
from semanticdist import utils

def get_single_entities(text, context):
    _params = {
        'text': text,
        'lang': 'AGNOSTIC',
        'annType': 'NAMED_ENTITIES',
        'key': context['babelfy_key']
    }
    resp = requests.post("https://babelfy.io/v1/disambiguate", data=_params).json()
    if "message" in resp:
        return None
    return resp

# We can only get some entities per day on API, so just get as many as we can
def get_entities(data, part, context, pickle_file=None):
    if f'{part}_entities' in data:
        needed_indexes = data.index[(~data[part].isnull()) & (data[f'{part}_entities'].isnull())]
    else:
        needed_indexes = data.index[~data[part].isnull()]
    for i in needed_indexes:
        entities = get_single_entities(data.loc[i, part], context)
        if entities is None:
            print("Semantic entity request failed.")
            break
        data[f'{part}_entities'] = entities
    if pickle_file is not None:
        utils.save_data(data, pickle_file, context)
    return data
