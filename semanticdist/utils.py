import pickle
import plotly.express as px
import numpy as np
import gspread_dataframe as gd
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def get_raw_data(context):
    _query = '''
        SELECT
            *
        FROM
            `moz-fx-data-shared-prod.regrets_reporter_analysis.yt_api_data_v7`
        WHERE
            takedown = FALSE
    '''
    data = context['bq_client'].query(
        _query
    ).result(
    ).to_dataframe(
        bqstorage_client=context['bq_storage_client']
    )
    total_rows = len(data)
    data.drop_duplicates(subset="video_id", keep='first',
                         inplace=True, ignore_index=True)
    unique_rows = len(data)
    if total_rows != unique_rows:
        print(
            f"Warning: raw table has {total_rows - unique_rows} duplicate rows or {100 * (total_rows - unique_rows) / unique_rows}%.")
    return data


def update_from_raw_data(data, context):
    _query = '''
        SELECT
            *
        FROM
            `moz-fx-data-shared-prod.regrets_reporter_analysis.yt_api_data_v7`
        WHERE
            takedown = FALSE
    '''
    new_data = context['bq_client'].query(
        _query
    ).result(
    ).to_dataframe(
        bqstorage_client=context['bq_storage_client']
    ).loc[lambda d: ~ d.video_id.isin(data.video_id)]
    if len(new_data) > 0:
        return pd.concat([data, new_data])
    else:
        print("Warning: no new data acquired.")
        return data


def save_data(data, pickle_file, context):
    with open(context['gdrive_path'] + pickle_file, 'wb') as handle:
        pickle.dump(data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


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

    for i, r in vvdata.iterrows():
        if r.id_a > r.id_b:
            temp = (r.title_a, r.channel_a, r.description_a, r.id_a)
            (r.title_a, r.channel_a, r.description_a, r.id_a) = (
                r.title_b, r.channel_b, r.description_b, r.id_b)
            (r.title_b, r.channel_b, r.description_b, r.id_b) = temp
    if existing != None:
        vvdata = vvdata[[(r.id_a, r.id_b) not in existing for i,
                         r in vvdata.iterrows()]]

    ss = context['gspread_client'].open("Videovote backend")
    try:
        ws = ss.add_worksheet(tab, rows=len(vvdata), cols="9")
    except Exception:
        ws = ss.worksheet(tab)
    gd.set_with_dataframe(ws, vvdata.reset_index(
        drop=True), include_index=False)


def init_eval_pickle(name, context):
    temp = {}
    with open(context['gdrive_path'] + name, 'wb') as handle:
        pickle.dump(temp, handle, protocol=pickle.HIGHEST_PROTOCOL)


def update_eval_data(eval_data, sheet, context):
    ws = context['gspread_client'].open("Videovote backend").worksheet(sheet)
    new_eval_data = gd.get_as_dataframe(ws).dropna(
        axis=1, how='all').dropna(how='all')
    for i, r in new_eval_data.iterrows():
        key = (r.id_a, r.id_b)
        if key in eval_data:
            eval_data[key] = eval_data[key] + [r.vote]
        else:
            eval_data[key] = [r.vote]
    return eval_data


def get_equality_matrix(data, part):
    d = pdist([[i] for i in data[part]], lambda x, y: 1 if x == y else 0)
    return squareform(d)


def print_data_diagnostics(data):
    n = len(data)
    print(f"Data is length {n}")
    nt = len(data[data.transcript.str.len() > 0])
    print(f"With transcripts: {nt}")
    possible_parts = ["title", "transcript", "description", "thumbnail"]
    possible_types = ["embedding", "entities"]
    dups = len(data[data.video_id.duplicated()])
    if dups != 0:
        print(f"Warning! {dups} dupes detected.")
    for part in possible_parts:
        ap_n = nt if part == "transcript" else n
        for type in possible_types:
            if f"{part}_{type}" in data:
                nv = data[f"{part}_{type}"].isnull().sum()
                print(
                    f"Data has {part}_{type} for {n-nv} rows or {(n-nv)/ap_n * 100}%")
