from semanticdist import utils
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(data, part, context, pickle_file=None):
    if f'{part}_embedding' in data:
        needed_indexes = data.index[(~data[part].isnull()) & (
            data[f'{part}_embedding'].isnull())]
    else:
        needed_indexes = data.index[~data[part].isnull()]
    texts = list(data.loc[needed_indexes, part])
    embeddings = list(context['language_model'].encode(
        texts, show_progress_bar=True))
    data.loc[needed_indexes, f'{part}_embedding'] = embeddings
    if pickle_file is not None:
        utils.save_data(data, pickle_file, context)
    return data


def get_similarity_matrix(embeddings):
    return cosine_similarity(list(embeddings))
