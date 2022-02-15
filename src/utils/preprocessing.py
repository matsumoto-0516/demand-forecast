'''
TODO
1. 時系列の単位が日ごと以外の場合への対応
2. カテゴリ値等の特徴量への対応
'''
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset


# 同じ日のデータを統合
def sum_data_same_day(
    data    : pd.DataFrame,
    ts      : str,
    target  : str,
    idcs    : List[str]
) -> pd.DataFrame:
    '''
    ex.
    (data_input)                     (data_output)
    | ts           target |          | ts           target |
    | 2010/01/01   1      |          | 2010/01/01   1      |
    | 2010/01/02   2      |    ->    | 2010/01/02   3      |
    | 2010/01/02   1      |
    '''
    groupby_keys = idcs.copy()
    groupby_keys.append(ts)
    data = data.groupby(groupby_keys)[target].sum().fillna(0).reset_index().sort_values(ts)
    return data


# 存在しない日付のデータを0で埋める
def fill_missing_date(
    data    : pd.DataFrame,
    ts      : str,
    target  : str,
    idcs    : List[str],
    start   : str,
    end     : str,
    freq    : str
) -> pd.DataFrame:
    '''
    ex.
    (data_input)                     (data_output)
    | ts           target |          | ts           target |
    | 2010/01/01   1      |          | 2010/01/01   1      |
    | 2010/01/03   2      |    ->    | 2010/01/02   0      |
    |                     |          | 2010/01/03   2      |
    '''
    # start～end の日付の連続値
    train_range = pd.DataFrame({ts:list(pd.date_range(start=start, end=end, freq=freq))})

    # データの存在しない日付のターゲット値を0で埋める
    data_list = []
    for id, d in data.groupby(idcs):
        data_tmp = pd.merge(train_range, d, on=ts, how='left')
        data_tmp[target] = data_tmp[target].fillna(0)
        data_tmp[idcs] = id
        data_list.append(data_tmp)
    data = pd.concat(data_list).reset_index(drop=True)

    return data


# トレーニングデータとテストデータの生成
def split_train_test(
    data                : pd.DataFrame,
    ts                  : str,
    idcs                : List[str],
    prediction_length   : int,
    roll_num            : int,
    distance            : int
) -> Tuple[pd.DataFrame, List]:

    end = data[ts].max()

    # トレーニングデータ
    train_data = []
    for id, d in data.groupby(idcs):
        delete_point = prediction_length + (roll_num - 1) * distance    # 何日分のデータを除くか
        split_date = end - timedelta(days=delete_point)                 # トレーニングに使用するデータの最終日の日付
        train_data.append(d[d[ts] <= split_date].sort_values(ts))       # トレーニングに使用するデータ
    train_data = pd.concat(train_data).reset_index(drop=True)           # 各idのデータを1つにまとめる

    # テストデータ
    test_data = []
    for i in range(roll_num):
        test_data_x = []
        test_data_y = []
        for id, d in data.groupby(idcs):
            delete_point = prediction_length + (roll_num - i - 1) * distance
            split_date_x = end - timedelta(days=delete_point)
            split_date_y = end - timedelta(days=delete_point - prediction_length)
            test_data_x.append(d[d[ts] <= split_date_x].sort_values(ts))                            # 予測を出力するためのデータ
            test_data_y.append(d[(d[ts] > split_date_x) & (d[ts] <= split_date_y)].sort_values(ts)) # 予測期間の真値

        test_data.append(
            {
                'test_x' : pd.concat(test_data_x).reset_index(drop=True),
                'test_y' : pd.concat(test_data_y).reset_index(drop=True),
            }
        )

    return train_data, test_data



# GluonTS形式のデータに変換
def GluonTSDataset(
    data                    : pd.DataFrame,
    target                  : str,
    idcs                    : List[str],
    start                   : str,
    freq                    : str,
    model_type_is_one_dim   : bool
) -> Tuple[pd.DataFrame, List]:

    # データをID毎に分割
    target_data = []
    id_list = []
    for id, d in data.groupby(idcs):
        target_data.append(d[target].values)
        id_list.append(id)
    target_data = np.array(target_data)

    # 単変量予測モデルの場合
    if model_type_is_one_dim:
        multi_series_with_ID_train = [{
            FieldName.ITEM_ID: i,
            FieldName.TARGET: target_data[i],
            FieldName.START: start,
            }
            for i in range(len(target_data))
        ]

    # 多変量予測モデルの場合
    else:
        multi_series_with_ID_train = [{
            FieldName.ITEM_ID: 0,
            FieldName.TARGET: target_data,
            FieldName.START: start,
            }
        ]

    data = ListDataset(
        multi_series_with_ID_train,
        freq = freq,
        one_dim_target=model_type_is_one_dim
    )

    return data, id_list