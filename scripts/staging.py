

from pyspark.sql import SparkSession
import vaex as vx
import gc
import os

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
        spark = SparkSession.builder \
            .master('local[*]') \
            .config("spark.driver.memory", "10g") \
            .appName('my-cool-app-mod') \
            .getOrCreate()
        print('Cargando universe')
        universe = spark.read.csv(abs_path + '/staging/universe.csv', header=True, sep='|')
        universe = vx.from_pandas(universe.toPandas(), copy_index=False)
        universe['key'] = universe['ID'] + '_' + universe['period']
        universe.head(5)
        print('Cargando movements')
        movements_preproc = spark.read.csv(abs_path + '/staging/movements_preproc.csv', header=True, sep='|')
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
