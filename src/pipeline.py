import pickle

import test_params              # 入力データが格納されているファイル
from utils import preprocessing # 前処理
from utils import model         # 学習モデル
from utils import evaluation    # 評価
import pandas as pd

if __name__ == '__main__':

    # パラメータ呼び出し
    param = test_params.test_case1
    input_dir = param['input_dir']
    target = param['target_col']
    ts = param['ts_col']
    idcs = param['id_col']
    start = param['start_date']
    end = param['end_date']
    freq = param['freq']
    hyperparams = param['hyperparams']
    hyperparams['freq'] = freq
    prediction_length = hyperparams['prediction_length']
    model_type_is_one_dim = True    # True:DepAR,DeepState    False:DeepVAR

    # 入力データ
    data = pd.read_csv(input_dir)
    data[ts] = pd.to_datetime(data[ts])


    # 前処理
    data = preprocessing.sum_data_same_day(data, ts, target, idcs)
    data = preprocessing.fill_missing_date(data, ts, target, idcs, start, end, freq)
    train_data, test_data = preprocessing.split_train_test(data, ts, idcs, prediction_length, roll_num=3, distance=2)
    train_data, id_list = preprocessing.GluonTSDataset(train_data, target, idcs, start, freq, model_type_is_one_dim)


    # 学習
    model_name = 'DeepAR'
    save_dir = 'Predictor'
    predictor = model.make_predictor(train_data, hyperparams, model_name, save_dir)

    # # 学習済みモデル読み込み(pickle)
    # with open(save_dir + '/' + model_name + '.pickle', mode='rb') as fp:
    #     predictor = pickle.load(fp)


    # 評価
    kpi = ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']
    evaluation.test(predictor, test_data, kpi, target, idcs, start, freq, model_type_is_one_dim)
