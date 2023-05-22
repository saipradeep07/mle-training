# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest(Using Grid search and Random Search)

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To install package
 Command: ```pip setup.py install```

 - Run nox to test Default paths after installation
 command: ```nox```

## To excute the scripts
To know about module usage and arguments they accept, use
    ```python $script_name -h```

All arguments are optional only, if not provided use default values configured.

 1. Data Fetching and Processing:
    - cd /src
    - python ingest_data.py --output_path $path_to_store_data_files

 2. Training Models:
    - cd /src
    - python train.py --input_path $input_data_path --output_path $path_to_save_model_objects

 3. Scoring Models:
    - cd /src
    - python score.py --models_path $path_to_model_object_files --data_path $path_to_data_files


## Folder Structure
```
 | Root_FOLDER
    | artifacts
        - Folder to save model objects
    | data
        | Raw
            - Raw data
        | train
            - train data
        | test
            - test data
    | docs
        | build
            - builded documentation files
        | source
            - Source files for doc
    | results
        - All results from the executions
    | src
        | mle_lib
            - common utilities library
        - scripts needed for execution
```
