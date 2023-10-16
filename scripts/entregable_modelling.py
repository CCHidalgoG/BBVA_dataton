from staging import load_universe #from console
# from scripts.staging import load_universe # from project
#from staging import load_universe
import numpy as np
import pandas as pd
import vaex as vx
from vaex.ml.xgboost import XGBoostModel
from sklearn.metrics import f1_score
import random
import os
import gc


def get_colnames_not_in_list(df, list_of_strings):
    """
  Obtiene los colnames de un PySpark DataFrame que no estén en una lista de dos strings.

  Args:
    df: El DataFrame de PySpark.
    list_of_strings: La lista de strings que se desean excluir.

  Returns:
    Una lista con los colnames que no están en la lista de strings.
  """

    colnames = df.get_column_names()
    return [colname for colname in colnames if colname not in list_of_strings]


def conv_to_float_fillna_0(df, cols):
    for col in cols:
        df[col] = df[col].astype('float64')

    for col in cols:
        mean_col = float(df[col].mean())
        df[col] = df[col].fillna(mean_col)
    return df


def equilibrate_sample(df, class_, N, with_diff_w=None):
    if with_diff_w is None:
        # En este caso cada clase es muestreada con N
        df2 = df.to_pandas_df()
        classes = df2[class_].value_counts().to_dict()
        most = N
        classes_list = []
        for key in classes.keys():
            classes_list.append(df2.loc[df2[class_] == key])
        classes_sample = []
        for i in range(1, len(classes_list)):
            classes_sample.append(classes_list[i].sample(most, replace=True))
        df_maybe = pd.concat(classes_sample)
        final_df = pd.concat([classes_list[0], df_maybe], axis=0)
        final_df = final_df.reset_index(drop=True)
        return vx.from_pandas(final_df)
    else:
        df2 = df.to_pandas_df()
        classes = df2[class_].value_counts().to_dict()
        # En este caso, N sería el total de clases y se reparte entre los pesos de cada clase
        assert len(with_diff_w) == len(classes.values()) - 1
        most = N
        classes_list = []
        for key in classes.keys():
            classes_list.append(df2.loc[df2[class_] == key])
        classes_sample = []
        for i in range(1, len(classes_list)):
            classes_sample.append(classes_list[i].sample(int(np.round(most*with_diff_w[i-1])), replace=True))
        df_maybe = pd.concat(classes_sample)
        final_df = pd.concat([classes_list[0], df_maybe], axis=0)
        final_df = final_df.reset_index(drop=True)
        return vx.from_pandas(final_df)


def submit(model, eval, alias, vars=['ID', 'period'], ruta='/home/chidalgo/git/BBVA_dataton/data/submit/'):
    predichos = list(model.predict(df=eval).astype('int32'))
    eval2 = eval.to_pandas_df()
    eval2 = eval2[vars]
    eval2['target'] = predichos
    eval2.to_csv(ruta + alias + '.csv', index=False)
    print('Submit guardado')
    return


def split_data(equil=True, n=30000, with_diff_w=None):
    random.seed(2020)
    universe = load_universe()
    cols_not_in_model = ['ID', 'period', 'partition', 'key']
    target = ['attrition']
    colnames_not_in_list = get_colnames_not_in_list(universe, cols_not_in_model)
    features = list(np.setdiff1d(colnames_not_in_list, target))
    # validation and train
    validation = universe[universe.partition == 'test'].drop('partition').drop('attrition')
    universe_preproc = universe[universe.partition == 'train'].drop('partition')
    # con to float
    validation = conv_to_float_fillna_0(validation, features)
    universe_preproc = conv_to_float_fillna_0(universe_preproc, colnames_not_in_list)
    universe_preproc = universe_preproc.dropna()
    universe_preproc['attrition'] = universe_preproc['attrition'].astype('int64')
    universe_preproc = universe_preproc.shuffle()
    universe_preproc['binary_target'] = universe_preproc.func.where(universe_preproc['attrition'] == 0, 0, 1)
    # Partition train-test
    df_train, df_test = universe_preproc.ml.train_test_split(test_size=0.2, verbose=True)
    if equil:
        df_train = equilibrate_sample(df_train, 'attrition', n, with_diff_w=with_diff_w)
    return df_train, df_test, features, validation


######################################XGBOOST##############################################
def xgboost_bin(df_train, df_test, features, lr=0.02):
    # First Phase
    print('Entrenamiento primera fase')
    ## xgboost binary class
    params = {'learning_rate': lr,
              'objective': 'binary:logistic',
              'random_state': 2020,
              'n_jobs': -1,
              'max_depth': 7,
              'reg_lambda': 5,
              'subsample': 0.8}

    booster_bin = XGBoostModel(features=features, target='binary_target',
                               num_boost_round=2800, params=params)
    booster_bin.fit(df=df_train, evals=[(df_train, 'train'), (df_test, 'test')], early_stopping_rounds=22,
                    verbose_eval=True)
    pred_xgb = booster_bin.predict(df=df_test).round()
    print('El F1-Score para la parte binaria es:',
          f1_score(y_true=df_test.binary_target.tolist(), y_pred=list(pred_xgb), average='macro'))
    return booster_bin


def xgboost_mult(df_train, df_test, features, lr=0.02):
    ## xgboost multiclass
    print('Entrenamiento segunda fase')
    df_train_mult = df_train[df_train['binary_target'] != 0]
    df_test_mult = df_test[df_test['binary_target'] != 0]
    df_train_mult['attrition'] = df_train_mult['attrition'] - 1
    df_test_mult['attrition'] = df_test_mult['attrition'] - 1

    params = {'learning_rate': lr,
              'num_class': 5,
              'random_state': 2020,
              'n_jobs': -1,
              'max_depth': 7,
              'reg_lambda': 5,
              'subsample': 0.8}

    booster_mult = XGBoostModel(features=features, target='attrition', num_boost_round=1400, params=params)
    booster_mult.fit(df=df_train_mult, evals=[(df_train_mult, 'train'), (df_test_mult, 'test')],
                     early_stopping_rounds=5,
                     verbose_eval=True)

    pred_xgb = booster_mult.predict(df=df_test_mult)
    print('El F1-Score para la parte multiclase es:',
          f1_score(y_true=df_test_mult.attrition.tolist(), y_pred=list(pred_xgb), average='macro'))

    return booster_mult


def xgboost_two_phases(df_train, df_test, features, lr=0.02):
    booster_bin = xgboost_bin(df_train, df_test, features, lr)
    booster_mult = xgboost_mult(df_train, df_test, features, lr)
    return booster_bin, booster_mult


def submit_xgb(booster_bin, booster_mult, validation, alias='xgb'):
    validation1 = validation.to_pandas_df()
    validation1['binary_target'] = list(booster_bin.predict(validation1).round())
    validation1['multiclass_target'] = list(booster_mult.predict(validation1))
    validation1['target'] = np.where(validation1['binary_target'] == 0, 0, validation1['multiclass_target']+1)
    print('Guardando submit')
    validation1[['ID', 'period', 'target']].to_csv(f'{os.getcwd()}/data/submit/{alias}.csv',
                                                   index=False)
    return validation1


######################################MAIN##############################################
def main():
    gc.collect()
    pesos = [0.6563560972894752, 0.11811932988256248, 0.0959119496855346, 0.08249846941615184, 0.04711415372627595]
    print('Preparación de datos')
    df_train, df_test, features, validation = split_data(equil=True, n=220000, with_diff_w=pesos)

    print("Entrenamiento de modelos\n")
    print("-----------------------Entrenando XGBOOST--------------------")
    gc.collect()
    booster_bin, booster = xgboost_two_phases(df_train, df_test, features, lr=0.05)
    # Guardando submit
    _ = submit_xgb(booster_bin, booster, validation, alias='xgb1')

    # Envío hacia kaggle
    print('Enviando a kaggle')
    ats = 'xgb1'
    try:
        os.system(f"kaggle competitions submit -c bbva-data-challenge-2023 -f {os.getcwd()}/data/submit/{ats}.csv -m 'gb1'")
    except ConnectionAbortedError:
        print('Envio no disponible')




if __name__ == '__main__':
    main()
