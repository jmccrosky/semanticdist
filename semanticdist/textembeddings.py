def get_embeddings(texts, context):
    return list(context['language_model'].encode(texts, show_progress_bar=True))