import requests

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

def get_entities(texts, context):
    return [get_single_entities(text, context) for text in texts]
