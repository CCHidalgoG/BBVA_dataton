import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import vaex as vx
import gc
import os



spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "12g") \
        .appName('my-cool-app') \
        .getOrCreate()


def conv_to_numeric(df, cols, fl = True):

    for col in cols:
        if fl:
            df = df.withColumn(col, df[col].cast('float').alias(col))
        else:
            df = df.withColumn(col, df[col].cast('int').alias(col))
    return df


def conv_yes_no_binary(df, cols):
    for col in cols:
        df = df.withColumn(col,
                           F.when(F.col(col)=='Yes', 1)
                           .otherwise(0)
                           )
    return df


def split_columns(df, cols):
    for col in cols:
        second_number = F.regexp_extract(df[col], r'_(\d+)', 1)
        df = df.withColumn(col, second_number)
    return df


def preprocessing():
    """
    En esta primera parte se cargan las tablas y se revisa su estructura. Este escript está dividido en:
    1. Cargue de bases de datos
    2. Procesamiento de bases de datos
    3. Unión de bases de datos
    4. Base de datos pre-modelo (output final)
    :return:
    """
    ################################### 1. Cargue de bases de datos ###################################

    abs_path = '/home/chidalgo/git/BBVA_dataton/data'
    # Cargamos la base de datos de balances y la revisamos en su estructura
    balances = spark.read.csv(abs_path + '/archive/balances.csv', header=True)
    balances = balances.withColumn('period2', F.col('period'))
    # ------------------
    bal_to_split = ['type', 'product', 'entity']
    balances = split_columns(balances, bal_to_split)
    balances = conv_to_numeric(balances, bal_to_split, fl=False)
    #-------
    # cols float:
    balances = conv_to_numeric(balances, ['balance_amount', 'days_default'] + bal_to_split)
    balances = balances.withColumn('month', balances['month'].cast('int').alias('month'))
    # One Hot Encoding
    # para la variable type
    balances = balances.withColumn('type1', F.when(balances.type == 1, 1).otherwise(0))
    balances = balances.withColumn('type2', F.when(balances.type == 2, 1).otherwise(0))
    balances = balances.withColumn('type3', F.when(balances.type == 3, 1).otherwise(0))
    balances = balances.withColumn('type4', F.when(balances.type == 4, 1).otherwise(0))
    balances = balances.withColumn('type5', F.when(balances.type == 5, 1).otherwise(0))
    balances = balances.withColumn('type7', F.when(balances.type == 7, 1).otherwise(0))
    # para la variable product
    balances = balances.withColumn('product1', F.when(balances.product == 1, 1).otherwise(0))
    balances = balances.withColumn('product2', F.when(balances.product == 2, 1).otherwise(0))
    balances = balances.withColumn('product3', F.when(balances.product == 3, 1).otherwise(0))
    balances = balances.withColumn('product4', F.when(balances.product == 4, 1).otherwise(0))
    balances = balances.withColumn('product5', F.when(balances.product == 5, 1).otherwise(0))
    balances = balances.withColumn('product6', F.when(balances.product == 6, 1).otherwise(0))
    balances = balances.withColumn('product7', F.when(balances.product == 7, 1).otherwise(0))
    # para la variable entity
    balances = balances.withColumn('entity1', F.when(balances.entity == 1, 1).otherwise(0))
    balances = balances.withColumn('entity2', F.when(balances.entity == 2, 1).otherwise(0))
    balances = balances.withColumn('entity3', F.when(balances.entity == 3, 1).otherwise(0))
    balances = balances.withColumn('entity4', F.when(balances.entity == 4, 1).otherwise(0))
    balances = balances.withColumn('entity5', F.when(balances.entity == 5, 1).otherwise(0))
    # Para Month
    balances = balances.withColumn('bal_month1', F.when(balances.month == 1, 1).otherwise(0))
    balances = balances.withColumn('bal_month2', F.when(balances.month == 2, 1).otherwise(0))
    balances = balances.withColumn('bal_month3', F.when(balances.month == 3, 1).otherwise(0))
    balances = balances.withColumn('bal_month4', F.when(balances.month == 4, 1).otherwise(0))
    balances = balances.withColumn('bal_month5', F.when(balances.month == 5, 1).otherwise(0))
    balances = balances.withColumn('bal_month6', F.when(balances.month == 6, 1).otherwise(0))
    balances = balances.withColumn('bal_month7', F.when(balances.month == 7, 1).otherwise(0))
    balances = balances.withColumn('bal_month8', F.when(balances.month == 8, 1).otherwise(0))
    balances = balances.withColumn('bal_month9', F.when(balances.month == 9, 1).otherwise(0))
    balances = balances.withColumn('bal_month10', F.when(balances.month == 10, 1).otherwise(0))
    balances = balances.withColumn('bal_month11', F.when(balances.month == 11, 1).otherwise(0))
    balances = balances.withColumn('bal_month12', F.when(balances.month == 12, 1).otherwise(0))


    balances_preproc = balances.groupby('ID', 'period').agg(
        # F.countDistinct('month').alias('count_month_balances'),
        # F.countDistinct('type').alias('count_type_balances'),
        # F.countDistinct('product').alias('count_product_balances'),
        # F.countDistinct('entity').alias('count_entity_balances'),
        F.mean('month').alias('mean_month_balances'),
        F.mean('type').alias('mean_type_balances'),
        F.mean('product').alias('mean_product_balances'),
        F.mean('entity').alias('mean_entity_balances'),
        F.mean('balance_amount').alias('mean_balance_amount_balances'),
        F.min('balance_amount').alias('min_balance_amount_balances'),
        F.max('balance_amount').alias('max_balance_amount_balances'),
        F.stddev('balance_amount').alias('std_balance_amount_balances'),
        #F.kurtosis('balance_amount').alias('kurt_balance_amount_balances'),
        F.sum('balance_amount').alias('sum_balance_amount_balances'),
        F.mean('days_default').alias('mean_days_default_balances'),
        F.min('days_default').alias('min_days_default_balances'),
        F.max('days_default').alias('max_days_default_balances'),
        F.stddev('days_default').alias('std_days_default_balances'),
        #F.kurtosis('days_default').alias('kurt_days_default_balances'),
        F.sum('days_default').alias('sum_days_default_balances'),
        F.max('type1').alias('type1'),
        F.max('type2').alias('type2'),
        F.max('type3').alias('type3'),
        F.max('type4').alias('type4'),
        F.max('type5').alias('type5'),
        F.max('type7').alias('type7'),
        F.max('product1').alias('product1'),
        F.max('product2').alias('product2'),
        F.max('product3').alias('product3'),
        F.max('product4').alias('product4'),
        F.max('product5').alias('product5'),
        F.max('product6').alias('product6'),
        F.max('product7').alias('product7'),
        F.max('entity1').alias('entity1'),
        F.max('entity2').alias('entity2'),
        F.max('entity3').alias('entity3'),
        F.max('entity4').alias('entity4'),
        F.max('entity5').alias('entity5'),
        F.max('bal_month1').alias('bal_month1'),
        F.max('bal_month2').alias('bal_month2'),
        F.max('bal_month3').alias('bal_month3'),
        F.max('bal_month4').alias('bal_month4'),
        F.max('bal_month5').alias('bal_month5'),
        F.max('bal_month6').alias('bal_month6'),
        F.max('bal_month7').alias('bal_month7'),
        F.max('bal_month8').alias('bal_month8'),
        F.max('bal_month9').alias('bal_month9'),
        F.max('bal_month10').alias('bal_month10'),
        F.max('bal_month11').alias('bal_month11'),
        F.max('bal_month12').alias('bal_month12')
        ).fillna(0)
    ########################## Cargamos la base de datos customers y procesamos ####################
    customers = spark.read.csv(abs_path + '/archive/customers.csv', header=True)
    #------------------
    custom_to_split = ['type_job', 'bureau_risk']
    customers = split_columns(customers, custom_to_split)
    #------------------
    custom_cols_to_bin = ['product_1', 'product_2', 'product_3', 'product_4', 'ofert_1', 'ofert_2', 'ofert_3']
    customers = conv_yes_no_binary(customers, custom_cols_to_bin)
    #------------------
    custom_cols_to_float = ['age', 'income',  'payroll']
    custom_to_int = ['time_from_specialized'] + custom_to_split
    customers = conv_to_numeric(customers, custom_cols_to_float)
    customers = conv_to_numeric(customers, custom_to_int, fl=False)
    # bureau risk
    customers = customers.withColumn('bureau_risk1', F.when(customers.bureau_risk == 1, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk2', F.when(customers.bureau_risk == 2, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk3', F.when(customers.bureau_risk == 3, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk4', F.when(customers.bureau_risk == 4, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk5', F.when(customers.bureau_risk == 5, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk6', F.when(customers.bureau_risk == 6, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk7', F.when(customers.bureau_risk == 7, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk8', F.when(customers.bureau_risk == 8, 1).otherwise(0))
    customers = customers.withColumn('bureau_risk9', F.when(customers.bureau_risk == 9, 1).otherwise(0))
    # type job
    customers = customers.withColumn('type_job1', F.when(customers.type_job == 1, 1).otherwise(0))
    customers = customers.withColumn('type_job2', F.when(customers.type_job == 2, 1).otherwise(0))
    customers = customers.withColumn('type_job3', F.when(customers.type_job == 3, 1).otherwise(0))
    customers = customers.withColumn('type_job4', F.when(customers.type_job == 4, 1).otherwise(0))
    customers = customers.withColumn('type_job5', F.when(customers.type_job == 5, 1).otherwise(0))
    customers = customers.withColumn('type_job6', F.when(customers.type_job == 6, 1).otherwise(0))
    customers = customers.withColumn('type_job7', F.when(customers.type_job == 7, 1).otherwise(0))
    # time from specialized
    customers = customers.withColumn('time_from_specialized1', F.when(customers.time_from_specialized == 0, 1).otherwise(0))
    customers = customers.withColumn('time_from_specialized1', F.when(customers.time_from_specialized == 1, 1).otherwise(0))
    customers = customers.withColumn('time_from_specialized1', F.when(customers.time_from_specialized == 2, 1).otherwise(0))
    customers = customers.withColumn('time_from_specialized1', F.when(customers.time_from_specialized == 3, 1).otherwise(0))
    customers = customers.withColumn('time_from_specialized1', F.when(customers.time_from_specialized == 4, 1).otherwise(0))
    customers = customers.withColumn('time_from_specialized1', F.when(customers.time_from_specialized == 5, 1).otherwise(0))
    customers = customers.drop('type_job').drop('bureau_risk').drop('time_from_specialized')
    # Cargamos la base liabilities y la visualizamos
    liabilities = spark.read.csv(abs_path + '/archive/liabilities.csv', header=True)
    liabilities = liabilities.withColumn('period2', F.col('period'))
    liabilities = split_columns(liabilities, ['period2'])
    liabilities = conv_to_numeric(liabilities, ['month', 'product_1', 'product_2', 'period2'])
    liabilities = liabilities.withColumn('month', liabilities['month'].cast('int').alias('month'))
    liabilities = liabilities.withColumn('liab_month1', F.when(liabilities.month == 1, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month2', F.when(liabilities.month == 2, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month3', F.when(liabilities.month == 3, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month4', F.when(liabilities.month == 4, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month5', F.when(liabilities.month == 5, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month6', F.when(liabilities.month == 6, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month7', F.when(liabilities.month == 7, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month8', F.when(liabilities.month == 8, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month9', F.when(liabilities.month == 9, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month10', F.when(liabilities.month == 10, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month11', F.when(liabilities.month == 11, 1).otherwise(0))
    liabilities = liabilities.withColumn('liab_month12', F.when(liabilities.month == 12, 1).otherwise(0))
    liabilities.show(5)

    ###########Procesamiento de liabilities:
    liabilities_proc = liabilities.groupby('ID', 'period').agg(
        F.count('month').alias('count_month_liabilities'),
        F.mean('product_1').alias('lbl_product_1_mean'),
        F.mean('product_2').alias('lbl_product_2_mean'),
        F.stddev('product_1').alias('lbl_product_1_std'),
        F.stddev('product_2').alias('lbl_product_2_std'),
        F.min('product_1').alias('lbl_product_1_min'),
        F.min('product_2').alias('lbl_product_2_min'),
        F.max('product_1').alias('lbl_product_1_max'),
        F.max('product_2').alias('lbl_product_2_max'),
        F.sum('product_1').alias('lbl_product_1_sum'),
        F.sum('product_2').alias('lbl_product_2_sum'),
        F.max('liab_month1').alias('liab_month1'),
        F.max('liab_month2').alias('liab_month2'),
        F.max('liab_month3').alias('liab_month3'),
        F.max('liab_month4').alias('liab_month4'),
        F.max('liab_month5').alias('liab_month5'),
        F.max('liab_month6').alias('liab_month6'),
        F.max('liab_month7').alias('liab_month7'),
        F.max('liab_month8').alias('liab_month8'),
        F.max('liab_month9').alias('liab_month9'),
        F.max('liab_month10').alias('liab_month10'),
        F.max('liab_month11').alias('liab_month11'),
        F.max('liab_month12').alias('liab_month12')
    ).fillna(0)
    # Se carga la base de datos movements y se visualiza
    movements = spark.read.csv(abs_path + '/bbva-data-challenge-2023/archive/movements.csv', header=True)
    movements = movements.withColumn('period2', F.col('period'))
    movements = split_columns(movements, ['period2'])
    movements = conv_to_numeric(movements, ['month', 'type_1', 'type_2', 'type_3', 'type_4', 'period2'])
    movements = movements.withColumn('month', movements['month'].cast('int').alias('month'))
    movements = movements.withColumn('move_month1', F.when(movements.month == 1, 1).otherwise(0))
    movements = movements.withColumn('move_month2', F.when(movements.month == 2, 1).otherwise(0))
    movements = movements.withColumn('move_month3', F.when(movements.month == 3, 1).otherwise(0))
    movements = movements.withColumn('move_month4', F.when(movements.month == 4, 1).otherwise(0))
    movements = movements.withColumn('move_month5', F.when(movements.month == 5, 1).otherwise(0))
    movements = movements.withColumn('move_month6', F.when(movements.month == 6, 1).otherwise(0))
    movements = movements.withColumn('move_month7', F.when(movements.month == 7, 1).otherwise(0))
    movements = movements.withColumn('move_month8', F.when(movements.month == 8, 1).otherwise(0))
    movements = movements.withColumn('move_month9', F.when(movements.month == 9, 1).otherwise(0))
    movements = movements.withColumn('move_month10', F.when(movements.month == 10, 1).otherwise(0))
    movements = movements.withColumn('move_month11', F.when(movements.month == 11, 1).otherwise(0))
    movements = movements.withColumn('move_month12', F.when(movements.month == 12, 1).otherwise(0))
    movements.show(5)

    ###########Procesamiento de movements:
    movements_preproc = movements.groupby('ID', 'period').agg(
        F.count('month').alias('count_month_movements'),
        F.mean('type_1').alias('mvm_type_1_mean'),
        F.mean('type_2').alias('mvm_type_2_mean'),
        F.mean('type_3').alias('mvm_type_3_mean'),
        F.mean('type_4').alias('mvm_type_4_mean'),
        F.stddev('type_1').alias('mvm_type_1_std'),
        F.stddev('type_2').alias('mvm_type_2_std'),
        F.stddev('type_3').alias('mvm_type_3_std'),
        F.stddev('type_4').alias('mvm_type_4_std'),
        F.min('type_1').alias('mvm_type_1_min'),
        F.min('type_2').alias('mvm_type_2_min'),
        F.min('type_3').alias('mvm_type_3_min'),
        F.min('type_4').alias('mvm_type_4_min'),
        F.max('type_1').alias('mvm_type_1_max'),
        F.max('type_2').alias('mvm_type_2_max'),
        F.max('type_3').alias('mvm_type_3_max'),
        F.max('type_4').alias('mvm_type_4_max'),
        F.kurtosis('type_1').alias('mvm_type_1_kurtosis'),
        F.kurtosis('type_2').alias('mvm_type_2_kurtosis'),
        F.kurtosis('type_3').alias('mvm_type_3_kurtosis'),
        F.kurtosis('type_4').alias('mvm_type_4_kurtosis'),
        F.sum('type_1').alias('mvm_type_1_sum'),
        F.sum('type_2').alias('mvm_type_2_sum'),
        F.sum('type_3').alias('mvm_type_3_sum'),
        F.sum('type_4').alias('mvm_type_4_sum'),
        F.max('move_month1').alias('move_month1'),
        F.max('move_month2').alias('move_month2'),
        F.max('move_month3').alias('move_month3'),
        F.max('move_month4').alias('move_month4'),
        F.max('move_month5').alias('move_month5'),
        F.max('move_month6').alias('move_month6'),
        F.max('move_month7').alias('move_month7'),
        F.max('move_month8').alias('move_month8'),
        F.max('move_month9').alias('move_month9'),
        F.max('move_month10').alias('move_month10'),
        F.max('move_month11').alias('move_month11'),
        F.max('move_month12').alias('move_month12')

    ).fillna(0)
    ### Digital #############3
    digital = spark.read.csv(abs_path + '/bbva-data-challenge-2023/digital.csv', header=True)
    conv_to_floatd = ['month', 'dig_1', 'dig_2', 'dig_3', 'dig_4', 'dig_5', 'dig_6', 'dig_7', 'dig_8', 'dig_9', 'dig_10', 'dig_11']
    digital = conv_to_numeric(digital, conv_to_floatd)
    digital = digital.withColumn('month', digital['month'].cast('int').alias('month'))
    digital = digital.withColumn('dig_month1', F.when(digital.month == 1, 1).otherwise(0))
    digital = digital.withColumn('dig_month2', F.when(digital.month == 2, 1).otherwise(0))
    digital = digital.withColumn('dig_month3', F.when(digital.month == 3, 1).otherwise(0))
    digital = digital.withColumn('dig_month4', F.when(digital.month == 4, 1).otherwise(0))
    digital = digital.withColumn('dig_month5', F.when(digital.month == 5, 1).otherwise(0))
    digital = digital.withColumn('dig_month6', F.when(digital.month == 6, 1).otherwise(0))
    digital = digital.withColumn('dig_month7', F.when(digital.month == 7, 1).otherwise(0))
    digital = digital.withColumn('dig_month8', F.when(digital.month == 8, 1).otherwise(0))
    digital = digital.withColumn('dig_month9', F.when(digital.month == 9, 1).otherwise(0))
    digital = digital.withColumn('dig_month10', F.when(digital.month == 10, 1).otherwise(0))
    digital = digital.withColumn('dig_month11', F.when(digital.month == 11, 1).otherwise(0))
    digital = digital.withColumn('dig_month12', F.when(digital.month == 12, 1).otherwise(0))
    digital.show(5)

    digital_proc = digital.groupby('ID', 'period').agg(
        F.mean('dig_1').alias('dig_1_mean'),
        F.mean('dig_2').alias('dig_2_mean'),
        F.mean('dig_3').alias('dig_3_mean'),
        F.mean('dig_4').alias('dig_4_mean'),
        F.mean('dig_5').alias('dig_5_mean'),
        F.mean('dig_6').alias('dig_6_mean'),
        F.mean('dig_7').alias('dig_7_mean'),
        F.mean('dig_8').alias('dig_8_mean'),
        F.mean('dig_9').alias('dig_9_mean'),
        F.mean('dig_10').alias('dig_10_mean'),
        F.mean('dig_11').alias('dig_11_mean'),
        #-----------std-----------
        F.stddev('dig_1').alias('dig_1_std'),
        F.stddev('dig_2').alias('dig_2_std'),
        F.stddev('dig_3').alias('dig_3_std'),
        F.stddev('dig_4').alias('dig_4_std'),
        F.stddev('dig_5').alias('dig_5_std'),
        F.stddev('dig_6').alias('dig_6_std'),
        F.stddev('dig_7').alias('dig_7_std'),
        F.stddev('dig_8').alias('dig_8_std'),
        F.stddev('dig_9').alias('dig_9_std'),
        F.stddev('dig_10').alias('dig_10_std'),
        F.stddev('dig_11').alias('dig_11_std'),
        #--------min-------------
        F.min('dig_1').alias('dig_1_min'),
        F.min('dig_2').alias('dig_2_min'),
        F.min('dig_3').alias('dig_3_min'),
        F.min('dig_4').alias('dig_4_min'),
        F.min('dig_5').alias('dig_5_min'),
        F.min('dig_6').alias('dig_6_min'),
        F.min('dig_7').alias('dig_7_min'),
        F.min('dig_8').alias('dig_8_min'),
        F.min('dig_9').alias('dig_9_min'),
        F.min('dig_10').alias('dig_10_min'),
        F.min('dig_11').alias('dig_11_min'),
        #----------------max-------------
        F.max('dig_1').alias('dig_1_max'),
        F.max('dig_2').alias('dig_2_max'),
        F.max('dig_3').alias('dig_3_max'),
        F.max('dig_4').alias('dig_4_max'),
        F.max('dig_5').alias('dig_5_max'),
        F.max('dig_6').alias('dig_6_max'),
        F.max('dig_7').alias('dig_7_max'),
        F.max('dig_8').alias('dig_8_max'),
        F.max('dig_9').alias('dig_9_max'),
        F.max('dig_10').alias('dig_10_max'),
        F.max('dig_11').alias('dig_11_max'),
        #-----------kurtosis-----------------
        F.kurtosis('dig_1').alias('dig_1_kurt'),
        F.kurtosis('dig_2').alias('dig_2_kurt'),
        F.kurtosis('dig_3').alias('dig_3_kurt'),
        F.kurtosis('dig_4').alias('dig_4_kurt'),
        F.kurtosis('dig_5').alias('dig_5_kurt'),
        F.kurtosis('dig_6').alias('dig_6_kurt'),
        F.kurtosis('dig_7').alias('dig_7_kurt'),
        F.kurtosis('dig_8').alias('dig_8_kurt'),
        F.kurtosis('dig_9').alias('dig_9_kurt'),
        F.kurtosis('dig_10').alias('dig_10_kurt'),
        F.kurtosis('dig_11').alias('dig_11_kurt'),
        #---------------sum----------------------
        F.sum('dig_1').alias('dig_1_sum'),
        F.sum('dig_2').alias('dig_2_sum'),
        F.sum('dig_3').alias('dig_3_sum'),
        F.sum('dig_4').alias('dig_4_sum'),
        F.sum('dig_5').alias('dig_5_sum'),
        F.sum('dig_6').alias('dig_6_sum'),
        F.sum('dig_7').alias('dig_7_sum'),
        F.sum('dig_8').alias('dig_8_sum'),
        F.sum('dig_9').alias('dig_9_sum'),
        F.sum('dig_10').alias('dig_10_sum'),
        F.sum('dig_11').alias('dig_11_sum'),
        F.max('dig_month1').alias('digi_month1'),
        F.max('dig_month2').alias('digi_month2'),
        F.max('dig_month3').alias('digi_month3'),
        F.max('dig_month4').alias('digi_month4'),
        F.max('dig_month5').alias('digi_month5'),
        F.max('dig_month6').alias('digi_month6'),
        F.max('dig_month7').alias('digi_month7'),
        F.max('dig_month8').alias('digi_month8'),
        F.max('dig_month9').alias('digi_month9'),
        F.max('dig_month10').alias('digi_month10'),
        F.max('dig_month11').alias('digi_month11'),
        F.max('dig_month12').alias('digi_month12')
    )

    # Cargamos universe train y visualizamos (añadimos etiqueta)
    universe_train = spark.read.csv(abs_path + '/bbva-data-challenge-2023/archive/universe_train.csv', header=True)
    universe_train = universe_train.withColumn("partition", F.lit("train"))
    universe_train.show(5)
    # Cargamos universe test, homologamos con train y visualizamos
    universe_test = spark.read.csv(abs_path + '/bbva-data-challenge-2023/archive/universe_test.csv', header=True)
    universe_test = universe_test.withColumn('attrition', F.lit(np.nan))
    universe_test = universe_test.withColumn('partition', F.lit("test"))
    universe_test.show(5)
    # unimos test y train
    universe = universe_train.unionByName(universe_test)
    universe.show(5)

    return universe, movements_preproc, liabilities_proc, balances_preproc, digital_proc, customers


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


def load_universe(abs_path='/home/chidalgo/git/BBVA_dataton/data'):
    if os.path.isfile(abs_path + '/preproc_full.hdf5'):
        universe = vx.open(abs_path + '/preproc_full.hdf5')
    else:
        universe, movements_preproc, liabilities_proc, balances_preproc, digital_proc, customers = preprocessing()
        print('Cargando universe')
        universe = vx.from_pandas(universe.toPandas(), copy_index=False)
        universe['key'] = universe['ID'] + '_' + universe['period']
        universe.head(5)
        print('Cargando movements')
        movements_preproc = vx.from_pandas(movements_preproc.toPandas(), copy_index=False)
        movements_preproc['key'] = movements_preproc['ID'] + '_' + movements_preproc['period']
        movements_preproc = movements_preproc.drop(['ID', 'period'])
        movements_preproc.head(5)
        print('Cargando liabilities')
        liabilities_proc = spark.read.csv(abs_path + '/staging/liabilities_proc.csv', header=True, sep='|')
        liabilities_proc = vx.from_pandas(liabilities_proc.toPandas(), copy_index=False)
        liabilities_proc['key'] = liabilities_proc['ID'] + '_' + liabilities_proc['period']
        liabilities_proc = liabilities_proc.drop(['ID', 'period'])
        liabilities_proc.head(5)
        print('Cargando balances')
        balances_preproc = spark.read.csv(abs_path + '/staging/balances_preproc.csv', header=True, sep='|')
        balances_preproc = vx.from_pandas(balances_preproc.toPandas(), copy_index=False)
        balances_preproc['key'] = balances_preproc['ID'] + '_' + balances_preproc['period']
        balances_preproc = balances_preproc.drop(['ID', 'period'])
        balances_preproc.head(5)
        print('Cargando digital')
        digital_proc = spark.read.csv(abs_path + '/staging/digital_proc.csv', header=True, sep='|')
        digital_proc = vx.from_pandas(digital_proc.toPandas(), copy_index=False)
        digital_proc['key'] = digital_proc['ID'] + '_' + digital_proc['period']
        digital_proc = digital_proc.drop(['ID', 'period'])
        digital_proc.head(5)
        print('Cargando customers')
        customers = spark.read.csv(abs_path + '/staging/customers.csv', header=True, sep='|')
        customers = vx.from_pandas(customers.toPandas(), copy_index=False)
        customers.head(5)
        spark.stop()
        gc.collect()
        # Unimos los dataframes anteriores con universe
        print('Realizando Joins')
        universe = (universe.join(movements_preproc, on='key', how='left').
                    join(liabilities_proc, on='key', how='left').
                    join(balances_preproc, on='key', how='left').
                    join(digital_proc, on='key', how='left').
                    join(customers, on='ID', how='left'))
        gc.collect()
        print('Guardando data')
        # universe.export_hdf5(abs_path + '/preproc_full.hdf5', progress='rich')
        universe.export(abs_path + '/preproc_full.hdf5', progress='rich')
    return universe


if __name__ == '__main__':
    load_universe()