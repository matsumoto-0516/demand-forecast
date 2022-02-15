from ast import Assert
import os
import pickle

from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer

# DeepAR学習器
def DeepAR(hyperparams : dict):
    default_hyperparams = {
        'freq' : 'D',
        'context_length' : hyperparams['prediction_length'],
        'num_layers' : 2,
        'num_cells' : 40,
        'dropout_rate' : 0.1,
        'batch_size' : 128,
        'use_feat_dynamic_real' : False,
        'use_feat_static_cat' : False,
        'cardinality' : None,
        'epochs' : 50,
        'learning_rate' : 1e-3
    }

    # 学習器を作成
    # 入力されたハイパーパラメータに該当するものがなければデフォルト値を使用
    estimator = DeepAREstimator(
        freq = hyperparams.get('freq', default_hyperparams['freq']),
        prediction_length = hyperparams['prediction_length'],
        context_length = hyperparams.get('context_length', default_hyperparams['context_length']),
        num_layers = hyperparams.get('num_layers', default_hyperparams['num_layers']),
        num_cells = hyperparams.get('num_cells', default_hyperparams['num_cells']),
        dropout_rate = hyperparams.get('dropout_rate', default_hyperparams['dropout_rate']),
        batch_size = hyperparams.get('batch_size', default_hyperparams['batch_size']),
        use_feat_dynamic_real = hyperparams.get('use_feat_dynamic_real', default_hyperparams['use_feat_dynamic_real']),
        use_feat_static_cat = hyperparams.get('use_feat_static_cat', default_hyperparams['use_feat_static_cat']),
        cardinality = hyperparams.get('cardinality', default_hyperparams['cardinality']),
        trainer=Trainer(
            epochs=hyperparams.get('epochs', default_hyperparams['epochs']),
            learning_rate=hyperparams.get('learning_rate', default_hyperparams['learning_rate'])
        )
    )

    return estimator

# DeepState学習器
def DeepState(hyperparams : dict):
    default_hyperparams = {
        'freq' : 'D',
        'num_layers' : 2,
        'num_cells' : 40,
        'dropout_rate' : 0.1,
        'batch_size' : 128,
        'use_feat_dynamic_real' : False,
        'use_feat_static_cat' : False,
        'cardinality' : [],
        'epochs' : 50,
        'learning_rate' : 1e-3
    }

    # 学習器を作成
    # 入力されたハイパーパラメータに該当するものがなければデフォルト値を使用
    estimator = DeepStateEstimator(
        freq = hyperparams.get('freq', default_hyperparams['freq']),
        prediction_length = hyperparams['prediction_length'],
        num_layers = hyperparams.get('num_layers', default_hyperparams['num_layers']),
        num_cells = hyperparams.get('num_cells', default_hyperparams['num_cells']),
        dropout_rate = hyperparams.get('dropout_rate', default_hyperparams['dropout_rate']),
        batch_size = hyperparams.get('batch_size', default_hyperparams['batch_size']),
        use_feat_dynamic_real = hyperparams.get('use_feat_dynamic_real', default_hyperparams['use_feat_dynamic_real']),
        use_feat_static_cat = hyperparams.get('use_feat_static_cat', default_hyperparams['use_feat_static_cat']),
        cardinality = hyperparams.get('cardinality', default_hyperparams['cardinality']),
        trainer=Trainer(
            epochs=hyperparams.get('epochs', default_hyperparams['epochs']),
            learning_rate=hyperparams.get('learning_rate', default_hyperparams['learning_rate'])
        )
    )

    return estimator

# DeepVAR学習器
def DeepVAR(hyperparams : dict, data : ListDataset):
    target_dim = len(data.list_data[0]['target'])
    default_hyperparams = {
        'freq' : 'D',
        'context_length' : hyperparams['prediction_length'],
        'num_layers' : 2,
        'num_cells' : 40,
        'dropout_rate' : 0.1,
        'batch_size' : 128,
        'cardinality' : [1],
        'epochs' : 50,
        'learning_rate' : 1e-3
    }

    # 学習器を作成
    # 入力されたハイパーパラメータに該当するものがなければデフォルト値を使用
    estimator = DeepVAREstimator(
        freq = hyperparams.get('freq', default_hyperparams['freq']),
        target_dim = target_dim,
        prediction_length = hyperparams['prediction_length'],
        context_length = hyperparams.get('context_length', default_hyperparams['context_length']),
        num_layers = hyperparams.get('num_layers', default_hyperparams['num_layers']),
        num_cells = hyperparams.get('num_cells', default_hyperparams['num_cells']),
        dropout_rate = hyperparams.get('dropout_rate', default_hyperparams['dropout_rate']),
        batch_size = hyperparams.get('batch_size', default_hyperparams['batch_size']),
        cardinality = hyperparams.get('cardinality', default_hyperparams['cardinality']),
        trainer=Trainer(
            epochs=hyperparams.get('epochs', default_hyperparams['epochs']),
            learning_rate=hyperparams.get('learning_rate', default_hyperparams['learning_rate'])
        )
    )

    return estimator

# 保存
def save_predictor(predictor, save_dir, model_name):
    # 出力先のディレクトリを作成
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + '/' + model_name + '.pickle', mode='wb') as fp:
        pickle.dump(predictor, fp)

# 学習
def make_predictor(train_data, hyperparams, model_name, save_dir):
    if model_name == 'DeepAR':
        estimator = DeepAR(hyperparams)
    elif model_name == 'DeepState':
        estimator = DeepState(hyperparams)
    elif model_name == 'DeepVAR':
        estimator = DeepVAR(hyperparams, train_data)



    predictor = estimator.train(training_data=train_data)
    save_predictor(predictor, save_dir, model_name)

    return predictor