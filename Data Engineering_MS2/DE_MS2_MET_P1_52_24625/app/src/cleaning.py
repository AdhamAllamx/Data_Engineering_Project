import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup



def load_csv(file_path):

    dataframe = pd.read_csv(file_path)
    print(f"> '{file_path}'.csv Dataframe loaded ")

    return dataframe

def get_file_paths():

    return {
        "LOOKUP_CSV": "data/lookup_fintech_data_MET_P1_52_24625.csv",
        "INT_RATE_MULTIVARIATE_LOOKUP_CSV": "data/int_rate_multivariate_lookup_fintech_data_MET_P1_52_24625.csv",
        "SCALING_LOOKUP_TABLE_CSV": "data/scaling_lookup_fintech_data_MET_P1_52_24625.csv",
        "ONE_HOT_ENCODING_LOOKUP_TABLE_CSV": "data/one_hot_lookup_fintech_data_MET_P1_52_24625.csv"
    }

def intialize_lookup_tables():

    imputation_lookup_df_int_rate = pd.DataFrame(columns=['feature', 'grade', 'imputed_value'])
    scaling_lookup_table = pd.DataFrame(columns=['column', 'min', 'max'])
    one_hot_encoding_lookup_table = pd.DataFrame(columns=['column','name_encoded_column'])
    lookup_table_df = pd.DataFrame(columns=['column', 'original', 'imputed'])

    #imputation_lookup_df_description = pd.DataFrame(columns=['feature', 'purpose', 'imputed_value'])
    # imputation_lookup_df_emp_title = pd.DataFrame(columns=['feature', 'annual_inc', 'imputed_value'])
    # imputation_lookup_df_emp_length = pd.DataFrame(columns=['feature', 'emp_title', 'imputed_value'])

    return lookup_table_df , imputation_lookup_df_int_rate,one_hot_encoding_lookup_table ,scaling_lookup_table


def load_all_lookup_tables():

    file_paths = get_file_paths()
    lookup_table = load_csv(file_paths["LOOKUP_CSV"])
    scaling_lookup_table = load_csv(file_paths["SCALING_LOOKUP_TABLE_CSV"])
    one_hot_encoding_lookup = load_csv(file_paths["ONE_HOT_ENCODING_LOOKUP_TABLE_CSV"])
    int_rate_multivariate_lookup = load_csv(file_paths["INT_RATE_MULTIVARIATE_LOOKUP_CSV"])

    return lookup_table ,int_rate_multivariate_lookup,one_hot_encoding_lookup,scaling_lookup_table

def create_dataframe_copy(df):
    print ("> Creating Copy from original Dataframe to be cleaned")
    return df.copy()



def standardize_column_names(df):
    print("> Standardize Dataframe Column Names")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def set_dataframe_index(df):
    print("> set Dataframe Index")
    df.set_index('customer_id', inplace=True)
    return df

def find_duplicates(df):
    print("> Handling Duplicates ")
    duplicates_df = df[df.duplicated()]
    return duplicates_df


def clean_and_replace_inconsistent_values_emp_title_column(df):
    print("> Cleaning and replacing inconsistent values in 'emp_title' feature")

    if 'emp_title' in df.columns:
        df['emp_title'] = df['emp_title'].apply(lambda x: str(x).lower().strip().replace(' ', '_') if pd.notnull(x) else x)

        replacements = {
            'maintenace': 'maintenance',
            'sales_and_man': 'sales_and_management',
            'licensed': 'licensed_vocational_nurse', 
            'motor_vehicle_rep': 'motor_vehicle_representative'
        }

        for original, new_value in replacements.items():
            df['emp_title'] = df['emp_title'].replace(original, new_value)

    return df




def clean_and_replace_inconsistent_values_type_column(df):
    print("> Cleaning and replacing inconsistent values in 'type' feature")
    if 'type' in df.columns:
        df['type'] = df['type'].apply(lambda x: str(x).lower().strip().replace(' ', '_') if pd.notnull(x) else x)

        replacements = {'joint_app': 'joint'}

        for original, new_value in replacements.items():
            df['type'] = df['type'].replace(original, new_value)

    return df


def clean_feature_str_lower_dashed_space(df, feature_name):
        print("> clean feature str lower case and replace space with underscore")
        df[feature_name] = df[feature_name].str.lower().str.strip().str.replace(' ', '_')
        return df



def add_to_multivariate_lookup(imputation_lookup_df, new_entry):
    new_entry_df = pd.DataFrame([new_entry],dtype="object")
    is_duplicate = (
        (imputation_lookup_df[list(new_entry)] == pd.Series(new_entry))
        .all(axis=1)
        .any()
    )

    if not is_duplicate and not new_entry_df.dropna(how='all').empty:
        imputation_lookup_df = pd.concat([imputation_lookup_df, new_entry_df], ignore_index=True)
    
    return imputation_lookup_df



def imputing_missing_values_annual_inc_joint(df, lookup_table_df, streaming=False):
    print("> Imputing missing values for 'annual_inc_joint'")
    
    if not streaming:
        df['annual_inc_joint'] = df['annual_inc_joint'].fillna(0)
        lookup_table_df = registery_mapping_lookup_table(lookup_table_df, 'annual_inc_joint', 'NaN', 0)
    else:
        encoding_map = lookup_table_df[lookup_table_df['column'] == "annual_inc_joint"].set_index('original')['imputed'].to_dict()
        imputed_value = encoding_map.get(np.nan)
        imputed_value = int(imputed_value) if isinstance(imputed_value, str) else imputed_value
        
        if  encoding_map:
            df['annual_inc_joint'] = df['annual_inc_joint'].fillna(imputed_value)
            print(f"> Streaming mode: Filled missing 'annual_inc_joint' with value '{imputed_value}'")
        else:
            print("> Warning: No imputed value found in lookup table for 'annual_inc_joint' in streaming mode.")
    
    return df, lookup_table_df


def imputing_missing_values_description(df, lookup_table_df, streaming=False):
    print("> Imputing missing values for 'description'")
    
    if not streaming:
        df['description'] = df['description'].fillna(df['purpose'])
        lookup_table_df = registery_mapping_lookup_table(
            lookup_table_df, 
            column='description', 
            original='NaN', 
            imputed='corresponding purpose value'
        )
    else:
            matched_row = lookup_table_df[
                (lookup_table_df['column'] == 'description') 
               
            ]
            df['description'] = df['description'].astype('object')
            if not matched_row.empty:

                df['description'] = df['description'].fillna(df['purpose'])
                print(f"> Streaming mode: Filled missing 'description' with corresponding 'purpose'")
            else:
                print("> Warning: No imputed value found in lookup table for 'description' in streaming mode.")
    
    return df, lookup_table_df



def imputing_missing_values_emp_title(df, lookup_table_df, streaming=False):
    print("> Imputing missing values for 'emp_title'")
    
    if not streaming:
        mode_value = df['emp_title'].mode()[0] if not df['emp_title'].mode().empty else 'Missing'
        df['emp_title'] = df['emp_title'].fillna(mode_value)
        lookup_table_df = registery_mapping_lookup_table(lookup_table_df, 'emp_title', 'NaN', mode_value+'(mode)')
        print(f"> Imputed missing 'emp_title' with mode value '{mode_value}'")
    else:
        encoding_map = lookup_table_df[lookup_table_df['column'] == "emp_title"].set_index('original')['imputed'].to_dict()
        imputed_value = encoding_map.get(np.nan)

        if imputed_value is not None:
            imputed_value = imputed_value.replace('(mode)', '').strip()
            df['emp_title'] = df['emp_title'].fillna(imputed_value)
            print(f"> Streaming mode: Filled missing 'emp_title' with value '{imputed_value}'")
        else:
            mode_value = df['emp_title'].mode()[0] if not df['emp_title'].mode().empty else 'Missing'
            df['emp_title'] = df['emp_title'].fillna(mode_value)
            print(f"> Warning: No imputed value found in lookup table for 'emp_title'. Falling back to mode value '{mode_value}'")
    
    return df, lookup_table_df




def imputing_missing_values_emp_length(df, lookup_table_df, streaming=False):
    print("> Imputing missing values for 'emp_length'")
    
    if not streaming:
        mode_value = df['emp_length'].mode()[0] if not df['emp_length'].mode().empty else 'Missing'
        df['emp_length'] = df['emp_length'].fillna(mode_value)
        lookup_table_df = registery_mapping_lookup_table(lookup_table_df, 'emp_length', 'NaN', mode_value+'(mode)')
        print(f"> Imputed missing 'emp_length' with mode value '{mode_value}'")
    else:
        encoding_map = lookup_table_df[lookup_table_df['column'] == "emp_length"].set_index('original')['imputed'].to_dict()
        imputed_value = encoding_map.get(np.nan)

        if imputed_value is not None:
            imputed_value = imputed_value.replace('(mode)', '').strip()
            df['emp_length'] = df['emp_length'].fillna(imputed_value)
            print(f"> Streaming mode: Filled missing 'emp_length' with value '{imputed_value}'")
        else:
            mode_value = df['emp_length'].mode()[0] if not df['emp_length'].mode().empty else 'Missing'
            df['emp_length'] = df['emp_length'].fillna(mode_value)
            print(f"> Warning: No imputed value found in lookup table for 'emp_length'. Falling back to mode value '{mode_value}'")
    
    return df, lookup_table_df



def imputing_missing_values_int_rate(df, imputation_lookup_df_int_rate, streaming=False):
    print("> Imputing missing values for 'int_rate'")
    
    if not streaming:
        missing_int_rate_mask = df['int_rate'].isna()
        mode_per_group = df.groupby('grade')['int_rate'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
        df.loc[missing_int_rate_mask, 'int_rate'] = mode_per_group[missing_int_rate_mask]
        
        unique_imputed_values = df[missing_int_rate_mask][['grade', 'int_rate']].dropna().drop_duplicates()
        if not unique_imputed_values.empty:
            for grade, int_rate in unique_imputed_values.values:
                new_entry = {'feature': 'int_rate', 'grade': float(grade), 'imputed_value': float(int_rate)}
                imputation_lookup_df_int_rate = add_to_multivariate_lookup(imputation_lookup_df_int_rate, new_entry)
    
    else:
        int_rate_lookup_dict = imputation_lookup_df_int_rate.set_index('grade')['imputed_value'].apply(float).to_dict()        
        missing_int_rate_mask = df['int_rate'].isna()
        if missing_int_rate_mask.any():
            df.loc[missing_int_rate_mask, 'int_rate'] = df.loc[missing_int_rate_mask, 'grade'].map(int_rate_lookup_dict)
            print(f"> Streaming mode: Filled missing 'int_rate' for grade ")
        else:
            print("> 'int_rate' is already present, no imputation needed")
    
    return df, imputation_lookup_df_int_rate


# this section of commented code contain the multivaraite imputation of emp_title , emp_length and description that not used in order to be generic as possible:


# def imputing_missing_values_description(df, imputation_lookup_df_description):
#     print("> imputing missing values description")
    
#     missing_description_mask = df['description'].isna()
#     mode_per_group = df.groupby('purpose')['description'].transform(lambda x: x.mode()[0] if not x.mode().empty else x.name)
#     df.loc[missing_description_mask, 'description'] = mode_per_group[missing_description_mask]

#     unique_purposes = df['purpose'].unique()
#     for purpose in unique_purposes:
#         description = df.loc[df['purpose'] == purpose, 'description'].mode()
#         description_value = description[0] if not description.empty else purpose.capitalize().replace("_", " ")
#         new_entry = {'feature': 'description', 'purpose': purpose, 'imputed_value': description_value}
#         imputation_lookup_df_description = add_to_multivariate_lookup(imputation_lookup_df_description, new_entry)

#     return df, imputation_lookup_df_description


# def imputing_missing_values_emp_title(df, imputation_lookup_df_emp_title):
#     print("> imputing missing values emp_title")
#     annual_income_col = 'annual_inc'
#     emp_title_col = 'emp_title'
    
#     income_ranges = df[df[emp_title_col].notna()].groupby(emp_title_col)[annual_income_col].agg(['min', 'max'])
#     income_ranges['avg'] = (income_ranges['min'] + income_ranges['max']) / 2
    
#     missing_emp_title_mask = df[emp_title_col].isna()
#     if missing_emp_title_mask.any():
#         diff = np.abs(df.loc[missing_emp_title_mask, annual_income_col].values[:, None] - income_ranges['avg'].values)
#         closest_idx = diff.argmin(axis=1)
#         closest_titles = income_ranges.iloc[closest_idx].index
        
#         df.loc[missing_emp_title_mask, emp_title_col] = pd.Series(closest_titles, index=df.loc[missing_emp_title_mask].index)
    
#         unique_imputed_values = df[missing_emp_title_mask][[annual_income_col, emp_title_col]].drop_duplicates()
#         for annual_inc, title in unique_imputed_values.values:
#             new_entry = {'feature': 'emp_title', 'annual_inc': annual_inc, 'imputed_value': title}
#             imputation_lookup_df_emp_title = add_to_multivariate_lookup(imputation_lookup_df_emp_title, new_entry)
    
#     return df, imputation_lookup_df_emp_title

# def imputing_missing_values_emp_title(df, imputation_lookup_df_emp_title):
#     print("> Imputing missing values for emp_title")
    
#     annual_income_col = 'annual_inc'
#     emp_title_col = 'emp_title'
#     # Step 1: Fill missing emp_title values by grouping by each unique annual_inc and using the mode of emp_title within that group
#     emp_title_mode_by_income = df.groupby(annual_income_col)[emp_title_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
#     missing_emp_title_mask = df[emp_title_col].isna()
#     df.loc[missing_emp_title_mask, emp_title_col] = df.loc[missing_emp_title_mask, annual_income_col].map(emp_title_mode_by_income)
#     # Step 2: Identify any remaining missing emp_title values
#     remaining_missing_mask = df[emp_title_col].isna()
#     # Step 3: For remaining missing values, define income bins and calculate mode within each bin
#     income_bin_width = 10000
#     income_bins = pd.cut(df[annual_income_col], bins=np.arange(0, df[annual_income_col].max() + income_bin_width, income_bin_width))
#     emp_title_mode_by_income_bin = df.groupby(income_bins, observed=False)[emp_title_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
#     df.loc[remaining_missing_mask, emp_title_col] = df.loc[remaining_missing_mask, annual_income_col].apply(
#         lambda x: emp_title_mode_by_income_bin[income_bins[df[annual_income_col] == x].values[0]]
#     )
#     remaining_missing_mask = df[emp_title_col].isna()
#     if remaining_missing_mask.any():
#         common_emp_title = df[emp_title_col].mode().iloc[0]
#         df.loc[remaining_missing_mask, emp_title_col] = common_emp_title
    
#     unique_imputed_values = df[missing_emp_title_mask][[annual_income_col, emp_title_col]].drop_duplicates()
#     for annual_inc, title in unique_imputed_values.values:
#         new_entry = {'feature': 'emp_title', 'annual_inc': annual_inc, 'imputed_value': title}
#         imputation_lookup_df_emp_title = add_to_multivariate_lookup(imputation_lookup_df_emp_title, new_entry)
#     remaining_after_imputation = df[emp_title_col].isna().sum()
#     print(f"Remaining missing values in 'emp_title' after all imputations: {remaining_after_imputation}")
    
#     return df, imputation_lookup_df_emp_title


# 

# def imputing_missing_values_emp_length(df, imputation_lookup_df_emp_length):
#     print("> Imputing missing values for emp_length")
    
#     mode_emp_length_by_title = df.groupby('emp_title')['emp_length'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    
#     missing_emp_length_mask = df['emp_length'].isna()
    
#     df.loc[missing_emp_length_mask, 'emp_length'] = mode_emp_length_by_title[missing_emp_length_mask]
    
#     overall_mode_emp_length = df['emp_length'].mode().iloc[0]
    
#     remaining_missing_mask = df['emp_length'].isna()
#     df.loc[remaining_missing_mask, 'emp_length'] = overall_mode_emp_length
    
#     unique_imputed_values = df[missing_emp_length_mask][['emp_title', 'emp_length']].dropna().drop_duplicates()
#     for title, emp_length in unique_imputed_values.values:
#         new_entry = {'feature': 'emp_length', 'emp_title': title, 'imputed_value': emp_length}
#         imputation_lookup_df_emp_length = add_to_multivariate_lookup(imputation_lookup_df_emp_length, new_entry)
#     remaining_after_imputation = df['emp_length'].isna().sum()
#     print(f"Remaining missing values in 'emp_length' after all imputations: {remaining_after_imputation}")
#     return df, imputation_lookup_df_emp_length




def map_emp_length_to_years_numeric(df ,lookup_table_df):
    print("> Mapping emp_length to years numeric")

    emp_length_mapping = {
        '10+ years': 11,
        '< 1 year': 0.5,
    }

    for i in range(1, 10):
        emp_length_mapping[f'{i} years'] = float(i)
    df['emp_length_years'] = df['emp_length'].map(emp_length_mapping)
    for original, imputed in emp_length_mapping.items():
        lookup_table_df = registery_mapping_lookup_table(lookup_table_df, 'emp_length', original, imputed)

    return df,lookup_table_df


def handling_outliers_log_transformation(df, column_name):
    print(f"> Handling Outliers for '{column_name}' using Log Transformation")
    if column_name == 'annual_inc_joint':
        new_column_name = f"{column_name}_log_transformation"
        df[new_column_name] = np.where(
            df['annual_inc_joint'] > 0,
            np.log1p(df['annual_inc_joint']),
            np.nan
        )
    else:
        new_column_name = f"{column_name}_log_transformation"
        df[new_column_name] = np.log1p(df[column_name])

    return df


def handling_outliers_sqrt_transformation(df, column_name):
    print(f"> Handling Outliers for '{column_name}' using Sqrt Transformation")
    new_column_name = f"{column_name}_sqrt_transformation"
    df[new_column_name] = np.sqrt(df[column_name])
    return df

def create_month_number_column(df):
    print("> Creating month_number column")
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['month_number'] = df['issue_date'].dt.month
    return df


def create_salary_can_cover_column(df):
    print("> Creating salary_can_cover column")
    df['salary_can_cover'] = (df['annual_inc'] >= df['loan_amount']).astype(int)
    return df

def create_letter_grade_column(df,lookup_table_df):
    print("> Creating letter_grade column")
    grade_mapping = {
        1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A',
        6: 'B', 7: 'B', 8: 'B', 9: 'B', 10: 'B',
        11: 'C', 12: 'C', 13: 'C', 14: 'C', 15: 'C',
        16: 'D', 17: 'D', 18: 'D', 19: 'D', 20: 'D',
        21: 'E', 22: 'E', 23: 'E', 24: 'E', 25: 'E',
        26: 'F', 27: 'F', 28: 'F', 29: 'F', 30: 'F',
        31: 'G', 32: 'G', 33: 'G', 34: 'G', 35: 'G'
    }
    df['letter_grade'] = df['grade'].map(grade_mapping)
    for original, imputed in grade_mapping.items():
        lookup_table_df = registery_mapping_lookup_table(lookup_table_df, 'grade', original, imputed)
    
    return df,lookup_table_df



def create_installment_per_month_column(df):
    def calculate_installment(row):
        P = row['loan_amount']
        r = row['int_rate'] / 12 / 100 
        term_value = str(row['term']).strip()  
        if term_value and term_value.split()[0].isdigit():
            n = int(term_value.split()[0]) 
        else:
            n = 1  
        if r == 0:  
            return P / n
        M = P * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        return M
    
    df['installment_per_month'] = df.apply(calculate_installment, axis=1)
    return df



def apply_one_hot_encoding(df, column_name, one_hot_encoding_lookup_table):
    print(f"> One-Hot Encoding for '{column_name}'")

    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, one_hot_encoded], axis=1)
    
    encoded_columns = list(one_hot_encoded.columns)
    for encoded_col in encoded_columns:
        one_hot_encoding_lookup_table = pd.concat([one_hot_encoding_lookup_table, pd.DataFrame({
            'column': [column_name],
            'name_encoded_column': [encoded_col]
        })], ignore_index=True)

    
    return df, one_hot_encoding_lookup_table

def apply_one_hot_encoding_for_streaming(df, column_name, one_hot_encoding_lookup_table):
    encoded_columns = one_hot_encoding_lookup_table[one_hot_encoding_lookup_table['column'] == column_name]['name_encoded_column'].tolist()
    for col in encoded_columns:
        df[col] = False  

    feature_value = df[column_name].values[0]
    matching_col = f"{column_name}_{feature_value}"

    if matching_col in encoded_columns:
        df[matching_col] = True  

    return df


def apply_label_encoding(df, column_name,lookup_table_df):
    print(f"> One Hot Encoding for '{column_name}'")
    encoder = LabelEncoder()
    new_column_name = f"{column_name}_label_encoded"
    df[new_column_name] = encoder.fit_transform(df[column_name])
    for original, encoded in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        lookup_table_df = registery_mapping_lookup_table(lookup_table_df, column_name, original, encoded)
    return df,lookup_table_df


def apply_label_encoding_for_streaming(df, column_name, lookup_df):
    original_value = df[column_name].iloc[0]
    
    lookup_match = lookup_df[
        (lookup_df["column"] == column_name) &
        (lookup_df["original"] == original_value)
    ]
    new_column_name = f"{column_name}_label_encoded"
    if not lookup_match.empty:
        imputed_value = lookup_match["imputed"].values[0]
        df[new_column_name] =imputed_value
        print(f"> Encoded '{column_name}' with value '{original_value}' to '{imputed_value}' using lookup table.")
    else:
        print(f"> No encoding found in lookup table for '{column_name}' with value '{original_value}'")
    
    return df


def apply_min_max_scaling(df, column_name, lookup_table_df):

    print(f"> Normalizing using MinMaxScaler for '{column_name}'")
    
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    
    scaler = MinMaxScaler()
    new_column_name = f"{column_name}_normalized"
    df[[new_column_name]] = scaler.fit_transform(df[[column_name]])
    
    lookup_table_entry = {
        'column': column_name,
        'min': min_value,
        'max': max_value
    }
    
    existing_entry = lookup_table_df[
        (lookup_table_df['column'] == column_name) & 
        (lookup_table_df['min'] == min_value)&
        (lookup_table_df['max']== max_value)
    ]
    
    if existing_entry.empty:
        lookup_table_df = pd.concat([lookup_table_df, pd.DataFrame([lookup_table_entry])], ignore_index=True)
        print(f"> Added scaling parameters for '{column_name}' to Scaling lookup table.")
    else:
        print(f"> Scaling parameters for '{column_name}' already exist in Scaling lookup table.")
    
    return df, lookup_table_df

def apply_min_max_scaling_for_streaming(row, column_name, lookup_table_df):
    print(f"> Normalizing using MinMaxScaler for '{column_name}'")
    scaling_params = lookup_table_df[lookup_table_df['column'] == column_name]
    if scaling_params.empty:
        print(f"> No scaling parameters found for column '{column_name}' in lookup table.")
        return row  

    min_value = scaling_params['min'].values[0]
    max_value = scaling_params['max'].values[0]

    column_value = row[column_name].iloc[0]  
    if pd.notna(column_value):
        normalized_value = (column_value - min_value) / (max_value - min_value)
        row[f"{column_name}_normalized"] = normalized_value
        print(f"> Applied Min-Max scaling to '{column_name}' for row with value {normalized_value}.")
    else:
        print(f"> Skipping scaling for '{column_name}' as value is NaN.")

    return row

def registery_mapping_lookup_table(lookup_table_df, column, original, imputed):
    new_entry = pd.DataFrame({
        'column': [column],
        'original': [original],
        'imputed': [imputed]
    })
    if not ((lookup_table_df['column'] == column) &
            (lookup_table_df['original'] == original) &
            (lookup_table_df['imputed'] == imputed)).any():
        lookup_table_df = pd.concat([lookup_table_df, new_entry], ignore_index=True)
    
    return lookup_table_df


def create_state_names_bonus(df):
    print("> Creating state_name column bonus")
    url = "https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if table is None:
            print("No table found on the page.")
            return df
        
        rows = table.find_all('tr')
        state_codes = []
        state_names = []
        
        for row in rows[1:]:
            columns = row.find_all('td')
            if len(columns) == 3:
                state_code = columns[2].text.strip()
                state_name = columns[0].text.strip()
                state_codes.append(state_code)
                state_names.append(state_name)
        
        state_df = pd.DataFrame({
            'state_code': state_codes,
            'state_name': state_names
        })
        state_dict = dict(zip(state_df['state_code'], state_df['state_name']))
        df['state_name'] = df['state'].map(state_dict)
        print("> State names added successfully.")
        
        return df
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return df
    

def drop_unnecessary_columns(df):
    print("> Dropping unnecessary columns ")
    columns_to_drop = [
        'avg_cur_bal', 'tot_cur_bal', 'loan_amount', 'funded_amount', 
        'int_rate', 'annual_inc', 'emp_length', 'grade',  'addr_state', 'purpose', 'home_ownership', 
        'verification_status', 'type', 'term', 'loan_status','state'
    ]  
    df = df.drop(columns=columns_to_drop,axis=1)
    return df


def data_preprocessing(df):

    df = create_dataframe_copy(df)
    df = standardize_column_names(df)
    df = set_dataframe_index(df)
    duplicates_df = find_duplicates(df)
    df = df.drop_duplicates()
    df = clean_and_replace_inconsistent_values_emp_title_column(df)
    df = clean_and_replace_inconsistent_values_type_column(df)

    return df 

def handling_missing_values_imputations(df, lookup_table_df,imputation_lookup_df_int_rate):

    df, lookup_table_df = imputing_missing_values_annual_inc_joint(df,lookup_table_df,False)
    df, lookup_table_df = imputing_missing_values_description(df, lookup_table_df,False)
    df, imputation_lookup_df_int_rate = imputing_missing_values_int_rate(df, imputation_lookup_df_int_rate,False)
    df, lookup_table_df = imputing_missing_values_emp_title(df,lookup_table_df,False)
    df, lookup_table_df = imputing_missing_values_emp_length(df,lookup_table_df,False)

    return df ,lookup_table_df ,imputation_lookup_df_int_rate

def handling_missing_values_imputations_streaming(df,lookup_table,int_rate_multivariate_lookup):

    df,lookup_table = imputing_missing_values_annual_inc_joint(df,lookup_table,True)
    df ,lookup_table= imputing_missing_values_description(df,lookup_table,True)
    df ,int_rate_multivariate_lookup= imputing_missing_values_int_rate(df, int_rate_multivariate_lookup,True)
    df,lookup_table= imputing_missing_values_emp_title(df, lookup_table,True)
    df,lookup_table = imputing_missing_values_emp_length(df, lookup_table,True)

    return df , lookup_table

def handling_outliers(df):

    df = handling_outliers_log_transformation(df, 'annual_inc')
    df = handling_outliers_sqrt_transformation(df, 'int_rate')
    df = handling_outliers_log_transformation(df, 'annual_inc_joint')
    df = handling_outliers_log_transformation(df, 'funded_amount')
    df = handling_outliers_log_transformation(df, 'loan_amount')
    df = handling_outliers_log_transformation(df, 'avg_cur_bal')
    df = handling_outliers_log_transformation(df, 'tot_cur_bal')

    return df

def feature_engineering(df , lookup_table_df):

    df = create_month_number_column(df)
    df = create_salary_can_cover_column(df)
    df,lookup_table_df = create_letter_grade_column(df,lookup_table_df)
    df = create_installment_per_month_column(df)

    return df , lookup_table_df

def encoding (df, lookup_table_df , one_hot_encoding_lookup_table):

    df,one_hot_encoding_lookup_table = apply_one_hot_encoding(df, 'home_ownership',one_hot_encoding_lookup_table)
    df,one_hot_encoding_lookup_table = apply_one_hot_encoding(df, 'verification_status',one_hot_encoding_lookup_table)
    df,one_hot_encoding_lookup_table = apply_one_hot_encoding(df, 'term',one_hot_encoding_lookup_table)
    df,one_hot_encoding_lookup_table = apply_one_hot_encoding(df, 'loan_status',one_hot_encoding_lookup_table)
    df,one_hot_encoding_lookup_table = apply_one_hot_encoding(df , 'type',one_hot_encoding_lookup_table)

    df,lookup_table_df= apply_label_encoding(df, 'addr_state',lookup_table_df)
    df,lookup_table_df = apply_label_encoding(df, 'state',lookup_table_df)
    df,lookup_table_df = apply_label_encoding(df, 'purpose',lookup_table_df)
    df,lookup_table_df= apply_label_encoding(df, 'letter_grade',lookup_table_df)

    return df , lookup_table_df ,one_hot_encoding_lookup_table

def encoding_streaming(df ,lookup_table ,one_hot_encoding_lookup):

    df = apply_one_hot_encoding_for_streaming(df, 'home_ownership',one_hot_encoding_lookup)
    df = apply_one_hot_encoding_for_streaming(df, 'verification_status',one_hot_encoding_lookup)
    df = apply_one_hot_encoding_for_streaming(df, 'term',one_hot_encoding_lookup)
    df = apply_one_hot_encoding_for_streaming(df, 'loan_status',one_hot_encoding_lookup)
    df = apply_one_hot_encoding_for_streaming(df , 'type',one_hot_encoding_lookup)

    df = apply_label_encoding_for_streaming(df, 'addr_state',lookup_table)
    df = apply_label_encoding_for_streaming(df, 'state',lookup_table)
    df = apply_label_encoding_for_streaming(df, 'purpose',lookup_table)
    df = apply_label_encoding_for_streaming(df, 'letter_grade',lookup_table)

    return df ,lookup_table

def normalization(df, scaling_lookup_table):

    df,scaling_lookup_table = apply_min_max_scaling(df, 'installment_per_month',scaling_lookup_table)
    df,scaling_lookup_table = apply_min_max_scaling(df, 'funded_amount_log_transformation',scaling_lookup_table)

    return df ,scaling_lookup_table

def normalization_streaming(df ,scaling_lookup_table):
    
    df = apply_min_max_scaling_for_streaming(df, 'installment_per_month',scaling_lookup_table)
    df = apply_min_max_scaling_for_streaming(df, 'funded_amount_log_transformation',scaling_lookup_table)

    return df


def clean(df):
    print("> Starting of cleaning the original Dataset")

    # Initialize individual DataFrames for imputation lookup tables
    lookup_table_df,imputation_lookup_df_int_rate,one_hot_encoding_lookup_table ,scaling_lookup_table =intialize_lookup_tables()
   
    # Step 1: Data Preprocessing :
    df = data_preprocessing(df)

    # Step 2: Imputing Missing Values :
    df,lookup_table_df,imputation_lookup_df_int_rate = handling_missing_values_imputations(df,lookup_table_df ,imputation_lookup_df_int_rate)

    # Step 3: Map employment length to numeric years
    df,lookup_table_df = map_emp_length_to_years_numeric(df ,lookup_table_df)

    # Step 4: Handle outliers with log and sqrt transformations
    df = handling_outliers(df)

    # Step 5: Feature engineering columns
    df,lookup_table_df= feature_engineering(df,lookup_table_df)

    # Step 6: Apply Encoding 
    df ,lookup_table_df,one_hot_encoding_lookup_table = encoding(df , lookup_table_df , one_hot_encoding_lookup_table)
    
    # Step 7: Apply Normalization 
    df ,scaling_lookup_table = normalization(df ,scaling_lookup_table)

    # Step 8: Optionally add state names using external source (bonus step)
    df = create_state_names_bonus(df)

    # Step 9: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Return the cleaned DataFrame and the individual lookup tables
    return df,lookup_table_df, imputation_lookup_df_int_rate,scaling_lookup_table,one_hot_encoding_lookup_table






def clean_stream_processing(df):
    print("> Starting cleaning streamed row from producer.")
    # Load all lookup tables needed for streaming 
    lookup_table ,int_rate_multivariate_lookup,one_hot_encoding_lookup,scaling_lookup_table = load_all_lookup_tables()

    # Step 1: Data Preprocessing :
    df = data_preprocessing(df)

    # Step 2: Imputing Missing Values :
    df ,lookup_table = handling_missing_values_imputations_streaming(df , lookup_table ,int_rate_multivariate_lookup)

    # Step 3: Map employment length to numeric years
    df ,lookup_table= map_emp_length_to_years_numeric(df,lookup_table)

    # Step 10: Handle outliers with log and sqrt transformations
    df = handling_outliers(df)

    # Step 5: Feature engineering columns
    df, lookup_table = feature_engineering(df,lookup_table)

    # Step 6: Apply Encoding 
    df , lookup_table= encoding_streaming(df , lookup_table, one_hot_encoding_lookup)

    # Step 7: Apply Normalization 
    df = normalization_streaming(df , scaling_lookup_table)

    # Step 15: Optionally add state names using external source (bonus step)
    df = create_state_names_bonus(df)

    # Step 16: Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Return the cleaned DataFrame and the individual lookup tables
    return df ,lookup_table











