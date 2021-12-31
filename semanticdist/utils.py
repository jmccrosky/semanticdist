import pickle

def get_raw_data(context):
    _query = '''
        SELECT
            *
        FROM
            `moz-fx-data-shared-prod.regrets_reporter_analysis.yt_api_data_v7`
        WHERE
            transcript != ''
            AND takedown = FALSE
    '''
    return context['bq_client'].query(
        _query
    ).result(
    ).to_dataframe(
        bqstorage_client=context['bq_storage_client']
    )

def save_data(data, pickle_file, context):
    with open(context['gdrive_path'] + pickle_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)