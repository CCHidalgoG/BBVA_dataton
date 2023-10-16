from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import vaex as vx


def get_colnames_not_in_list(df, list_of_strings):
    """
  Obtiene los colnames de un PySpark DataFrame que no estén en una lista de dos strings.

  Args:
    df: El DataFrame de PySpark.
    list_of_strings: La lista de strings que se desean excluir.

  Returns:
    Una lista con los colnames que no están en la lista de strings.
  """

    colnames = df.columns
    return [colname for colname in colnames if colname not in list_of_strings]


def load_universe():
    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "10g") \
        .appName('my-cool-app-mod') \
        .getOrCreate()
    abs_path = '/home/chidalgo/git/BBVA_dataton/data'
    print('Cargando universe')
    universe = spark.read.csv(abs_path + '/staging/universe.csv', header=True, sep='|')
    universe.show(5)
    print('Cargando movements')
    movements_preproc = spark.read.csv(abs_path + '/staging/movements_preproc.csv', header=True, sep='|')
    movements_preproc.show(5)
    print('Cargando liabilities')
    liabilities_proc = spark.read.csv(abs_path + '/staging/liabilities_proc.csv', header=True, sep='|')
    liabilities_proc.show(5)
    print('Cargando balances')
    balances_preproc = spark.read.csv(abs_path + '/staging/balances_preproc.csv', header=True, sep='|')
    balances_preproc.show(5)
    print('Cargando digital')
    digital_proc = spark.read.csv(abs_path + '/staging/digital_proc.csv', header=True, sep='|')
    digital_proc.show(5)
    print('Cargando customers')
    customers = spark.read.csv(abs_path + '/staging/customers.csv', header=True, sep='|')
    customers.show(5)
    # Unimos los dataframes anteriores con universe
    universe = (universe.join(movements_preproc, on=['ID', 'period'], how='left').
                join(liabilities_proc, on=['ID', 'period'], how='left').
                join(balances_preproc, on=['ID', 'period'], how='left').
                join(digital_proc, on=['ID', 'period'], how='left').
                join(customers, on=['ID']))
    return universe


def pos_process_data():
    abs_path = '/home/chidalgo/git/BBVA_dataton/data'
    # Columns not in model
    # ToDo: Añadir un split a period y obtener el número como un input más
    universe = load_universe()
    cols_not_in_model = ['ID', 'period', 'partition']

    colnames_not_in_list = get_colnames_not_in_list(universe, cols_not_in_model)

    for col in colnames_not_in_list:
        universe = universe.withColumn(col, universe[col].cast('float').alias(col))
    # Separamos la data de nuevo
    validation = universe.where(universe['partition'] == 'test').drop('partition').fillna(0)

    universe_preproc = universe.where(universe['partition'] == 'train').drop('partition').fillna(0)

    # Transformamos los datos
    assembler = VectorAssembler(inputCols=colnames_not_in_list, outputCol='features')

    transformed_data = assembler.transform(universe_preproc)
    validation = assembler.transform(validation)

    # Partición
    (training_data, test_data) = transformed_data.randomSplit([0.8, 0.2], seed=2020)
    print("Training Dataset Count: " + str(training_data.count()))
    print("Test Dataset Count: " + str(test_data.count()))

    #########################GBTC#######################################

    rf = RandomForestClassifier(labelCol='attrition', featuresCol='features',
                                  maxMemoryInMB=10000, seed=2020)
    model = rf.fit(training_data)
    rf_predictions = model.transform(test_data)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol='attrition', metricName='precisionByLabel')
    print('Random Forest classifier F1-Score:', multi_evaluator.evaluate(rf_predictions))

    ### SUBMIT RF
    rf_submit = model.transform(validation)
    (rf_submit[['ID', 'period', 'attrition']].withColumnRenamed('attrition', 'target').
     toPandas().to_csv(abs_path + '/submit/' + 'rf_submit.csv', header=True, index=False))

def prepross_obsolete()

    # Meses



    #######################################LIGHTGBM#########################################
    def lgbm_bin(df_train, df_test, features):
        target = 'binary_target'

        lgbooster = lightgbm.LGBMClassifier(num_leaves=5, max_depth=5, n_estimators=100, random_state=2020)

        # Make it a vaex transformer (for the automagic pipeline and lazy predictions)
        model = vaex.ml.sklearn.Predictor(features=features,
                                          target=target,
                                          model=lgbooster,
                                          prediction_name='prediction')
        model.fit(df_train)

        pred_xgb = model.predict(df=df_test).round()
        print('El F1-Score para la parte binaria es:',
              f1_score(y_true=df_test.binary_target.tolist(), y_pred=list(pred_xgb), average='macro'))
        return model

    def lgbm_mult(df_train, df_test, features):
        target = 'attrition'

        lgbooster_m = lightgbm.LGBMClassifier(num_leaves=5, max_depth=5, n_estimators=100, random_state=2020)

        # Make it a vaex transformer (for the automagic pipeline and lazy predictions)
        model = vaex.ml.sklearn.Predictor(features=features,
                                          target=target,
                                          model=lgbooster_m,
                                          prediction_name='prediction')
        model.fit(df_train)

        pred_xgb = model.predict(df=df_test).round()
        print('El F1-Score para la parte multi clase es:',
              f1_score(y_true=df_test.attrition.tolist(), y_pred=list(pred_xgb), average='macro'))
        return model

    def submit_lgbm(lgbm_bin, lgbm_mult, validation):
        validation1 = validation.to_pandas_df()
        validation1['binary_target'] = [lgbm_bin.transform(validation)['prediction']]
        validation1['multiclass_target'] = np.array(lgbm_mult.transform(validation)['prediction']).tolist()
        validation1['target'] = np.where(validation1['binary_target'] == 0, 0, validation1['multiclass_target'])
        print('Guardando submit')
        validation1[['ID', 'period', 'target']].to_csv('/home/chidalgo/git/BBVA_dataton/data/submit/lgbm.csv',
                                                       index=False)
        return validation1

    ###############################LIGHTGBM###########################
    lgbm_bin_m = lgbm_bin(df_train, df_test, features)
    lgbm_mult_m = lgbm_mult(df_train, df_test, features)
    submit_lgbm(lgbm_bin_m, lgbm_mult_m, validation)

if __name__ == '__main__':
    pos_process_data()
