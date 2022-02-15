from typing import Dict, List
import pandas as pd
import numpy as np

from utils import preprocessing # 前処理

# 平均絶対誤差
def MAE(y_true, y_pred):
    return np.sum(y_true - y_pred) / len(y_true)

# 平均二乗誤差
def MSE(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)

# 平均二乗誤差の平方根
def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

# 平均絶対パーセント誤差：真値が0でないもののみで計算
def MAPE(y_true, y_pred):
    mape = 0
    for true, pred in zip(y_true, y_pred):
        if true != 0:
            mape += np.abs((true - pred)) / true
    return mape / np.count_nonzero(true)

# 対称平均絶対パーセント誤差
def SMAPE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) / len(y_true)


def test(
    predictor,
    test_data               : List,
    kpi_list                : List[str],
    target                  : str,
    idcs                    : List[str],
    start                   : str,
    freq                    : str,
    model_type_is_one_dim   : bool
) -> None:

    for data in test_data:
        # 予測のためのデータを準備
        data_x = data['test_x']
        data_x, id_list = preprocessing.GluonTSDataset(
            data_x, target, idcs, start, freq, model_type_is_one_dim=model_type_is_one_dim
        )
        forecast_list = list(predictor.predict(data_x))   # 各id毎の予測結果をリストで取得

        # 真値
        true_list = []
        test_y = data['test_y']
        for id, d in test_y.groupby(idcs):
            true_list.append(d[target].values)

        # kpiを出力
        for kpi in kpi_list:
            result = []
            for forecast, true in zip(forecast_list, true_list):
                if kpi == 'MAE':
                    result.append(MAE(y_true=true, y_pred=forecast.mean))
                elif kpi == 'MSE':
                    result.append(MSE(y_true=true, y_pred=forecast.mean))
                elif kpi == 'RMSE':
                    result.append(RMSE(y_true=true, y_pred=forecast.mean))
                elif kpi == 'MAPE':
                    result.append(MAPE(y_true=true, y_pred=forecast.mean))
                elif kpi == 'SMAPE':
                    result.append(SMAPE(y_true=true, y_pred=forecast.mean))

            print('{}：{:.3f}'.format(kpi, np.mean(result)))

