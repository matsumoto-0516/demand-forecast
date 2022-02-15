test_case1 = {
    'input_dir' : 'Data/sales.csv',
    'ts_col' : 'date',
    'target_col' : 'sales',
    'id_col' : ['store'],
    'freq' : 'D',
    'start_date' : '2013-01-01',
    'end_date' : '2017-12-31',
    'hyperparams' : {
        'prediction_length' : 30,
        'context_length' : 30,
        'epochs' : 1,
        'num_layers' : 2,
        'num_cells' : 40
    }
}

test_case2 = {
    'input_dir' : 'Data/sales.csv',
    'ts_col' : 'date',
    'target_col' : 'sales',
    'id_col' : ['store', 'item'],
    'freq' : 'D',
    'start_date' : '2013-01-01',
    'end_date' : '2017-12-31',
    'hyperparams' : {
        'prediction_length' : 30,
        'context_length' : 30,
        'epochs' : 25,
        'num_layers' : 2,
        'num_cells' : 40
    }
}