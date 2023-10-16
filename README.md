# BBVA_dataton
This repository manage the code used in BBVA dataton 2023

I used two files:
* entregable_preprocessing.py: This file contains all ETL processes applied to diferents data files.
* entregable_modelling.py: This file contains all modelling steps to generate submission.csv

Two files can run from terminal, if you want, can put into a terminal:

```shell
python entregable_modelling.py
```

This activate the preprocessing script and after runs the model.

I have the next project structure:

```
BBVA_dataton
    data
        archive
            balances.csv
            customers.csv
            liabilities.csv
        bbva-data-challenge-2023
            archive
                movements.csv
                sample_submission.csv
                universe_test.csv
                universe_train.csv
            digital.csv
        submit
            xgb1.csv
    scripts
        entregable_modelling.py
        entregable_preprocessing.py
    requirements.txt
    README.md
```

Hint: Into entregable_preprocessing.py you can change the ram capacity when load pyspark:

```python
spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "12g") \
        .appName('my-cool-app') \
        .getOrCreate()
```

In this case I used 12g if RAM, but in a machine with 32g it can be change for 28g.
