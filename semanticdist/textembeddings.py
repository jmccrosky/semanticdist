from semanticdist import utils

def get_embeddings(data, part, context, pickle_file=None):
    texts = list(data[part])
    embeddings = list(context['language_model'].encode(texts, show_progress_bar=True))
    data[f'{part}_embedding'] = embeddings
    if pickle_file is not None:
        utils.save_data(data, pickle_file, context)
    return data