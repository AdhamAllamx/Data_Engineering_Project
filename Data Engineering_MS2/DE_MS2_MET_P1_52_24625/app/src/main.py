print('> Starting...')

import os
import pandas as pd
from cleaning import clean , load_csv
from db import  save_to_db
from consumer import consume_kafka_stream ,initialize_kafka_conumer
from run_producer import start_producer, stop_container
import time



def get_file_paths():
    return {
        "DATA_PATH": "data/fintech_data_2_52_24625.csv",
        "CLEANED_CSV": "data/fintech_data_MET_P1_52_24625_clean.csv",
        "LOOKUP_CSV": "data/lookup_fintech_data_MET_P1_52_24625.csv",
        "DESCRIPTION_MULTIVARIATE_LOOKUP_CSV": "data/description_multivariate_lookup_fintech_data_MET_P1_52_24625.csv",
        "INT_RATE_MULTIVARIATE_LOOKUP_CSV": "data/int_rate_multivariate_lookup_fintech_data_MET_P1_52_24625.csv",
        "SCALING_LOOKUP_TABLE_CSV": "data/scaling_lookup_fintech_data_MET_P1_52_24625.csv",
        "ONE_HOT_LOOKUP_TABLE_CSV": "data/one_hot_lookup_fintech_data_MET_P1_52_24625.csv",
        "EMP_TITLE_MULTIVARIATE_LOOKUP_CSV": "data/emp_title_multivariate_lookup_fintech_data_MET_P1_52_24625.csv",
        "EMP_LENGTH_MULTIVARIATE_LOOKUP_CSV": "data/emp_length_multivariate_lookup_fintech_data_MET_P1_52_24625.csv"
    }

def get_table_names():
    return {
        "CLEANED_DATA_SET": "fintech_data_MET_P1_52_24625_clean",
        "LOOKUP": "lookup_fintech_data_MET_P1_52_24625",
        "INT_RATE_MULTIVARIATE_LOOKUP": "int_rate_multivariate_lookup_fintech_data_MET_P1_52_24625",
        "SCALING_LOOKUP_TABLE": "scaling_lookup_fintech_data_MET_P1_52_24625",
        "ONE_HOT_LOOKUP_TABLE": "one_hot_lookup_fintech_data_MET_P1_52_24625"
    }


def load_csv_with_default_na_values(file_path):
    print("> Load CSV Dataframe")
    na_values = ['NA', 'Missing', 'NaN', '', ' ', 'null', 'None', 'N/A', 'n/a', 'UNKNOWN', 'unknown', 'undefined']
    df = pd.read_csv(file_path, na_values=na_values)
    return df

def save_csv_cleaned_df(df, filename):
    df.to_csv(filename, index=True)
    print(f"> Saved DataFrame to {filename}")

def save_csv_lookup_tables(df, filename):
    df.to_csv(filename, index=False)
    print(f"> Saved DataFrame to {filename}")

def main():
    file_paths = get_file_paths()
    table_names = get_table_names()
    
    if os.path.exists(file_paths["CLEANED_CSV"]):
        # Load existing CSV files
        cleaned_df = load_csv(file_paths["CLEANED_CSV"])
        lookup_df = load_csv(file_paths["LOOKUP_CSV"])
        imputation_lookup_df_int_rate = load_csv(file_paths["INT_RATE_MULTIVARIATE_LOOKUP_CSV"])
        scaling_lookup_table = load_csv(file_paths["SCALING_LOOKUP_TABLE_CSV"])
        one_hot_lookup_table = load_csv(file_paths["ONE_HOT_LOOKUP_TABLE_CSV"])
        
        # Save data to database tables
        save_to_db(cleaned_df, table_names["CLEANED_DATA_SET"])
        save_to_db(lookup_df, table_names["LOOKUP"])
        save_to_db(imputation_lookup_df_int_rate, table_names["INT_RATE_MULTIVARIATE_LOOKUP"])
        save_to_db(scaling_lookup_table, table_names["SCALING_LOOKUP_TABLE"])
        save_to_db(one_hot_lookup_table, table_names["ONE_HOT_LOOKUP_TABLE"])
    else:
        # Load initial data and clean it
        df = load_csv_with_default_na_values(file_paths["DATA_PATH"])

        cleaned_df, lookup_df, imputation_lookup_df_int_rate, scaling_lookup_table, one_hot_lookup_table = clean(df)
        
        # Save cleaned data to CSV files
        save_csv_cleaned_df(cleaned_df, file_paths["CLEANED_CSV"])
        save_csv_lookup_tables(lookup_df, file_paths["LOOKUP_CSV"])
        save_csv_lookup_tables(imputation_lookup_df_int_rate, file_paths["INT_RATE_MULTIVARIATE_LOOKUP_CSV"])
        save_csv_lookup_tables(scaling_lookup_table, file_paths["SCALING_LOOKUP_TABLE_CSV"])
        save_csv_lookup_tables(one_hot_lookup_table, file_paths["ONE_HOT_LOOKUP_TABLE_CSV"])

        # Save data to database tables
        save_to_db(cleaned_df, table_names["CLEANED_DATA_SET"])
        save_to_db(lookup_df, table_names["LOOKUP"])
        save_to_db(imputation_lookup_df_int_rate, table_names["INT_RATE_MULTIVARIATE_LOOKUP"])
        save_to_db(scaling_lookup_table, table_names["SCALING_LOOKUP_TABLE"])
        save_to_db(one_hot_lookup_table, table_names["ONE_HOT_LOOKUP_TABLE"])
        # save_to_db(imputation_lookup_df_description, 'imputation_lookup_df_description')
        # save_to_db(imputation_lookup_df_emp_length, 'imputation_lookup_df_emp_length')
        # save_to_db(imputation_lookup_df_emp_title, 'imputation_lookup_df_emp_title')
        # save_csv(imputation_lookup_df_description,DESCRIPTION_MULTIVARIATE_LOOKUP_CSV)

    # Initialize Kafka Consumer 
    consumer =initialize_kafka_conumer()
    # Start Kafka Producer
    id = start_producer('52_24625', 'kafka:9092', 'de_ms2_52_24625')
    # Start Kafka Consumer streaming 
    consume_kafka_stream(consumer)

    # Stop the producer container after consuming
    stop_container(id)

    print('> Done!')
    



if __name__ == "__main__":
    main()
