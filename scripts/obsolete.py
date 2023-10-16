import numpy as np
import pandas as pd
import vaex
from pyspark.shell import spark


def simplified_vars_quant(df, cols_to_trans, cols_to_group, cols_to_count, alias_db):
    df2 = pd.DataFrame()
    df2_vx = vaex.from_pandas(df2)
    for col in cols_to_trans:
        df2_vx[f'{alias_db}_{col}_mean'] = np.asanyarray(df.groupby(cols_to_group).agg(F.mean(col)).toPandas().iloc[:, 1].tolist())
        df2_vx[f'{alias_db}_{col}_stv'] = np.asanyarray(df.groupby(cols_to_group).agg(F.stddev_samp(col)).toPandas().iloc[:, 1].tolist())
        df2_vx[f'{alias_db}_{col}_min'] = np.asanyarray(df.groupby(cols_to_group).agg(F.min(col)).toPandas().iloc[:, 1].tolist())
        df2_vx[f'{alias_db}_{col}_max'] = np.asanyarray(df.groupby(cols_to_group).agg(F.max(col)).toPandas().iloc[:, 1].tolist())
        df2_vx[f'{alias_db}_{col}_kurtosis'] = np.asanyarray(df.groupby(cols_to_group).agg(F.kurtosis(col)).toPandas().iloc[:, 1].tolist())
        df2_vx[f'{alias_db}_{col}_sum'] = np.asanyarray(df.groupby(cols_to_group).agg(F.sum(col)).toPandas().iloc[:, 1].tolist())

    if cols_to_count is not None:
        for col2 in cols_to_count:
            df2_vx[f'{alias_db}_{col2}_count'] = np.asanyarray(df.groupby(cols_to_group).agg(F.count(col2)).toPandas().iloc[:, 1].tolist())
        df2 = df2_vx.to_pandas_df().dropna(axis=1, how='all')
        # iteritems is removed from pandas 2.0
        df2.iteritems = df2.items
        df2_spark = spark.createDataFrame(df2)
        return df2_spark
    else:
        df2 = df2_vx.to_pandas_df().dropna(axis=1, how='all')
        # iteritems is removed from pandas 2.0
        df2.iteritems = df2.items
        df2_spark = spark.createDataFrame(df2)
        return df2_spark

        # -----------------------------
    preproc_balances = simplified_vars_quant(df=balances,
                                             cols_to_trans=['balance_amount', 'days_default'] + bal_to_split,
                                             cols_to_group=['ID', 'period'],
                                             cols_to_count=['month'],
                                             alias_db='bal')


    preproc_customer = simplified_vars_quant(df=customers,
                                             cols_to_trans=custom_cols_to_float + custom_cols_to_bin,
                                             cols_to_group=['ID'],
                                             cols_to_count=None,
                                             alias_db='ctm')

    #########################TRAIN RANDOM FOREST#######################################

    rf = RandomForestClassifier(labelCol='attrition', featuresCol='features', maxMemoryInMB=11000)
    model = rf.fit(training_data)
    rf_predictions = model.transform(test_data)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol='attrition', metricName='precisionByLabel')
    print('Random Forest classifier F1-Score:', multi_evaluator.evaluate(rf_predictions))

    ### SUBMIT RF
    rf_submit = model.transform(validation)
    (rf_submit[['ID', 'period', 'attrition']].withColumnRenamed('attrition', 'target').
     toPandas().to_csv(abs_path + '/submit/' + 'rf_submit.csv', header=True, index=False))

    ######################################CATBOOST##############################################
    def catboost_two_phases(df_train, df_test, features, lr=0.05, md=5):
        # First Phase
        print('Entrenamiento primera fase')
        params = {
            'leaf_estimation_method': 'Gradient',
            'learning_rate': lr,
            'max_depth': md,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'sampling_frequency': 'PerTree',
            'colsample_bylevel': 0.8,
            'reg_lambda': 0.8,
            'objective': 'Logloss',
            'eval_metric': 'F1',
            'random_state': 2020,
            'verbose': 2,
        }
        ## catboost binary class
        catboos_bin = CatBoostModel(features=features, target='binary_target',
                                    params=params, prediction_type='Class')
        catboos_bin.fit(df=df_train, evals=[df_train, df_test], early_stopping_rounds=5)

        pred_cat = catboos_bin.predict(df=df_test).round()
        print('El F1-Score para la parte binaria es:',
              f1_score(y_true=df_test.binary_target.tolist(), y_pred=list(pred_cat), average='macro'))

        ## catboost multiclass
        print('Entrenamiento segunda fase')
        df_train_mult = df_train[df_train['binary_target'] != 0]
        df_test_mult = df_test[df_test['binary_target'] != 0]
        df_train_mult['attrition'] = df_train_mult['attrition'] - 1
        df_test_mult['attrition'] = df_test_mult['attrition'] - 1

        params = {
            'leaf_estimation_method': 'Gradient',
            'learning_rate': lr,
            'max_depth': md,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'sampling_frequency': 'PerTree',
            'colsample_bylevel': 0.8,
            'reg_lambda': 0.8,
            'objective': 'MultiClass',
            'custom_metric': 'MultiClass',
            'random_state': 2020,
            'verbose': 2,
        }
        booster_cat2 = CatBoostModel(features=features, target='attrition',
                                     params=params, prediction_type='Class')
        booster_cat2.fit(df=df_train_mult, evals=[df_train_mult, df_test_mult], early_stopping_rounds=5)

        pred_cat2 = [x[0] for x in booster_cat2.predict(df=df_test_mult)]
        print('El F1-Score para la parte multiclase es:',
              f1_score(y_true=df_test_mult.attrition.tolist(), y_pred=pred_cat2, average='macro'))
        return catboos_bin, booster_cat2

    def submit_cbst(catboos_bin, booster_cat2, validation):
        validation1 = validation.to_pandas_df()
        validation1['binary_target'] = list(catboos_bin.predict(validation1).round())
        validation1['multiclass_target'] = [x[0] + 1 for x in booster_cat2.predict(df=validation1)]
        validation1['target'] = np.where(validation1['binary_target'] == 0, 0, validation1['multiclass_target'])
        print('Guardando submit')
        validation1[['ID', 'period', 'target']].to_csv('/home/chidalgo/git/BBVA_dataton/data/submit/catboost.csv',
                                                       index=False)
        return validation1

    print("----------------------Entrenando CATBOOST--------------------")
    catboos_bin, booster_cat2 = catboost_two_phases(df_train, df_test, features, lr=0.05, md=5)
    cat_val = submit_cbst(catboos_bin, booster_cat2, validation)

    # Ensamble
    print('Entrenando ensamble')
    df_test_ensamble = df_test.to_pandas_df()
    df_test_ensamble['binary_target_xgb'] = list(booster_bin.predict(df_test).round())
    df_test_ensamble['multiclass_target_xgb'] = list((booster.predict(df_test) + 1.0).astype('int32'))
    df_test_ensamble['binary_target_cat'] = list(catboos_bin.predict(df_test).round())
    df_test_ensamble['multi_class_target_cat'] = [x[0] for x in booster_cat2.predict(df=df_test)]
    df_ensamble = pd.DataFrame({'pred_xgb': np.where(df_test_ensamble['binary_target_xgb'] == 0, 0,
                                                     df_test_ensamble['multiclass_target_xgb']),
                                'pred_cat': np.where(df_test_ensamble['binary_target_cat'] == 0, 0,
                                                     df_test_ensamble['multi_class_target_cat']),
                                'real': df_test_ensamble.attrition})

    df_ensamble.to_csv('/home/chidalgo/git/BBVA_dataton/data/real_predict/ensamble.csv', index=False)

    regr = linear_model.LinearRegression()
    X_train = np.array(df_ensamble[['pred_xgb', 'pred_cat']])
    y_train = df_ensamble['real'].values
    # Entrenamos nuestro modelo
    regr.fit(X_train, y_train)
    # Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
    y_pred = regr.predict(X_train).round()
    print(f1_score(y_true=y_train, y_pred=y_pred, average='macro'))
    print('Independent term: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    y_pred_val = regr.predict(pd.DataFrame({'pred_xgb': xgb_val.target, 'pred_cat': cat_val.target})).round()
    reg_sub = pd.DataFrame({'ID': xgb_val.ID, 'period': xgb_val.period, 'target': y_pred_val})
    reg_sub.to_csv('/home/chidalgo/git/BBVA_dataton/data/submit/lm.csv', index=False)