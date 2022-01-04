import pickle
import plotly.express as px
import numpy as np
import gspread_dataframe as gd
import pandas as pd


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


def load_data(pickle_file, context):
    with open(context['gdrive_path'] + pickle_file, 'rb') as handle:
        return pickle.load(handle)


def plot_similarity_matrix(m):
    fig = px.imshow(m, width=1600, height=800)

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )

    return fig


def get_indices_of_k_largest(arr, k):
    arr[np.tril_indices(arr.shape[0], 0)] = np.nan
    idx = np.argpartition((-arr).ravel(), k)
    return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])


def prep_videovote_sheet(data, pairs, tab, context, existing=None):
    left = data.iloc[pairs[0]].reset_index()
    right = data.iloc[pairs[1]].reset_index()

    vvdata = pd.DataFrame({
        "title_a": left.title,
        "channel_a": left.channel,
        "description_a": left.description,
        "id_a": left.video_id,
        "title_b": right.title,
        "channel_b": right.channel,
        "description_b": right.description,
        "id_b": right.video_id,
        "vote": None,

    })

    vvdata[['id_a','id_b']]=np.sort(vvdata[['id_a','id_b']].values,axis=1)
    if existing != None:
        vvdata = vvdata[(vvdata.id_a, vvdata.id_a) not in existing]

    ss = context['gspread_client'].open("Videovote backend")
    try:
        ws = ss.add_worksheet(tab, rows=len(vvdata), cols="9")
    except Exception:
        pass
    gd.set_with_dataframe(ws, vvdata.reset_index(
        drop=True), include_index=False)

def init_eval_pickle(name, context):
    temp = {}
    with open(context['gdrive_path'] + name, 'wb') as handle:
        pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)