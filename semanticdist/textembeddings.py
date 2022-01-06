from semanticdist import utils
from sklearn.metrics.pairwise import cosine_similarity


def get_embeddings(data, part, context, pickle_file=None):
    count_before = 0
    if f'{part}_embedding' in data:
        needed_indexes = data.index[(~data[part].isnull()) & (
            data[f'{part}_embedding'].isnull())]
        count_before = len(data[~data[f'{part}_embedding'].isnull()])
    else:
        needed_indexes = data.index[~data[part].isnull()]
        data[f'{part}_embedding'] = None
    texts = list(data.loc[needed_indexes, part])
    embeddings = list(context['language_model'].encode(
        texts, show_progress_bar=True))
    for i in range(len(needed_indexes)):
      data.at[needed_indexes[i], f'{part}_embedding'] = embeddings[i]
    count_after = len(data[~data[f'{part}_embedding'].isnull()])
    if count_after != count_before + len(needed_indexes):
        print(f"Warning: counts are wrong.  Pickle not saved. {count_after} {count_before} {len(needed_indexes)}")
        return data
    if pickle_file is not None:
        utils.save_data(data, pickle_file, context)
    return data


def get_similarity_matrix(embeddings):
    return cosine_similarity(list(embeddings))
