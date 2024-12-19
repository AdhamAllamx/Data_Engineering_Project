import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np



def extract_clean(filename):
    df = pd.read_csv(filename)
    df = clean(df)
    df.to_csv('/opt/airflow/data/fintech_clean.csv',index=False)
    print('loaded after cleaning succesfully')

def transform(filename):
    df = pd.read_csv(filename)
    df = apply_transformations(df)
    try:
        df.to_csv('/opt/airflow/data/fintech_transformed.csv',index=False, mode='x')
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')


def load_to_db(filename): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/fintech_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'fintech_etl_52_24625',con = engine,if_exists='replace')



def clean(df):
    df = data_preprocessing(df)
    df = handling_missing_values_imputations(df)
    return df 


def apply_transformations(df):
    df = map_emp_length_to_years_numeric(df)
    df = handling_outliers_transformations(df)
    df = feature_engineering(df)
    df = encoding(df)
    df = normalization(df)
    # df= drop_unnecessary_columns(df)

    return df 




#############################################
def data_preprocessing(df):

    df = create_dataframe_copy(df)
    df = standardize_column_names(df)
    df = set_dataframe_index(df)
    duplicates_df = find_duplicates(df)
    df = df.drop_duplicates()
    df = clean_and_replace_inconsistent_values_emp_title_column(df)
    df = clean_and_replace_inconsistent_values_type_column(df)

    return df 

#############################################

def handling_missing_values_imputations(df):

    df = imputing_missing_values_annual_inc_joint(df)
    df = imputing_missing_values_description(df)
    df = imputing_missing_values_int_rate(df)
    df = imputing_missing_values_emp_title(df)
    df = imputing_missing_values_emp_length(df)


    return df 


#############################################

def handling_outliers_transformations(df):

    df = handling_outliers_log_transformation(df, 'annual_inc')
    df = handling_outliers_sqrt_transformation(df, 'int_rate')
    df = handling_outliers_log_transformation(df, 'annual_inc_joint')
    df = handling_outliers_log_transformation(df, 'funded_amount')
    df = handling_outliers_log_transformation(df, 'loan_amount')
    df = handling_outliers_log_transformation(df, 'avg_cur_bal')
    df = handling_outliers_log_transformation(df, 'tot_cur_bal')

    return df

#############################################

def feature_engineering(df ):

    df = create_month_number_column(df)
    df = create_salary_can_cover_column(df)
    df = create_letter_grade_column(df)
    df = create_installment_per_month_column(df)

    return df 

#############################################

def encoding (df):

    df = apply_one_hot_encoding(df, 'home_ownership')
    df = apply_one_hot_encoding(df, 'verification_status')
    df = apply_one_hot_encoding(df, 'term')
    df = apply_one_hot_encoding(df, 'loan_status')
    df = apply_one_hot_encoding(df , 'type')

    df= apply_label_encoding(df, 'addr_state')
    df = apply_label_encoding(df, 'state')
    df = apply_label_encoding(df, 'purpose')
    df= apply_label_encoding(df, 'letter_grade')

    return df  


#############################################

def normalization(df):

    df = apply_min_max_scaling(df, 'installment_per_month')
    df = apply_min_max_scaling(df, 'funded_amount_log_transformation')

    return df 




## Data Cleaning & Preprocessing Functions 

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



def imputing_missing_values_annual_inc_joint(df):
    print("> Imputing missing values for 'annual_inc_joint'")
    df['annual_inc_joint'] = df['annual_inc_joint'].fillna(0)
    return df



def imputing_missing_values_emp_title(df):
    print("> Imputing missing values for 'emp_title'")
    annual_income_col = 'annual_inc'
    emp_title_col = 'emp_title'
    emp_title_mode_by_income = df.groupby(annual_income_col)[emp_title_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    missing_emp_title_mask = df[emp_title_col].isna()
    df.loc[missing_emp_title_mask, emp_title_col] = df.loc[missing_emp_title_mask, annual_income_col].map(emp_title_mode_by_income)
    remaining_missing_mask = df[emp_title_col].isna()
    income_bin_width = 10000
    income_bins = pd.cut(df[annual_income_col], bins=np.arange(0, df[annual_income_col].max() + income_bin_width, income_bin_width))
    emp_title_mode_by_income_bin = df.groupby(income_bins, observed=False)[emp_title_col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    
    df.loc[remaining_missing_mask, emp_title_col] = df.loc[remaining_missing_mask, annual_income_col].apply(
        lambda x: emp_title_mode_by_income_bin[income_bins[df[annual_income_col] == x].values[0]]
    )
    remaining_missing_mask = df[emp_title_col].isna()
    if remaining_missing_mask.any():
        common_emp_title = df[emp_title_col].mode().iloc[0]
        df.loc[remaining_missing_mask, emp_title_col] = common_emp_title
    mode_value = df['emp_title'].mode()[0] if not df['emp_title'].mode().empty else 'Missing'

    remaining_after_imputation = df[emp_title_col].isna().sum()
    print(f"Remaining missing values in 'emp_title' after all imputations: {remaining_after_imputation}")

    return df





def imputing_missing_values_emp_length(df):
    print("> Imputing missing values for 'emp_length'")
    
    mode_emp_length_by_title = df.groupby('emp_title')['emp_length'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    
    missing_emp_length_mask = df['emp_length'].isna()
    
    df.loc[missing_emp_length_mask, 'emp_length'] = mode_emp_length_by_title[missing_emp_length_mask]
    
    overall_mode_emp_length = df['emp_length'].mode().iloc[0]
    
    remaining_missing_mask = df['emp_length'].isna()
    df.loc[remaining_missing_mask, 'emp_length'] = overall_mode_emp_length

    remaining_after_imputation = df['emp_length'].isna().sum()
    print(f"Remaining missing values in 'emp_length' after all imputations: {remaining_after_imputation}")
    mode_value = df['emp_length'].mode()[0] if not df['emp_length'].mode().empty else 'Missing'

    return df



def imputing_missing_values_int_rate(df):
    print("> Imputing missing values for 'int_rate'")
    missing_int_rate_mask = df['int_rate'].isna()
    mode_per_group = df.groupby('grade')['int_rate'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
    df.loc[missing_int_rate_mask, 'int_rate'] = mode_per_group[missing_int_rate_mask]
    return df




def imputing_missing_values_description(df):
    print("> imputing missing values description")
    missing_description_mask = df['description'].isna()
    mode_per_group = df.groupby('purpose')['description'].transform(lambda x: x.mode()[0] if not x.mode().empty else x.name)
    df.loc[missing_description_mask, 'description'] = mode_per_group[missing_description_mask]

    return df




## Transformations Functions 


def map_emp_length_to_years_numeric(df ):
    print("> Mapping emp_length to years numeric")

    emp_length_mapping = {
        '10+ years': 11,
        '< 1 year': 0.5,
    }

    for i in range(1, 10):
        emp_length_mapping[f'{i} years'] = float(i)
    df['emp_length_years'] = df['emp_length'].map(emp_length_mapping)
    return df


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

def create_letter_grade_column(df):
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
 
    
    return df


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



def apply_one_hot_encoding(df, column_name):
    print(f"> One-Hot Encoding for '{column_name}'")
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, one_hot_encoded], axis=1)
    return df



def apply_label_encoding(df,column_name):
    print(f"> One Hot Encoding for '{column_name}'")
    encoder = LabelEncoder()
    new_column_name = f"{column_name}_label_encoded"
    df[new_column_name] = encoder.fit_transform(df[column_name])
    return df


def apply_min_max_scaling(df, column_name):

    print(f"> Normalizing using MinMaxScaler for '{column_name}'")
    
    min_value = df[column_name].min()
    max_value = df[column_name].max()

    scaler = MinMaxScaler()
    new_column_name = f"{column_name}_normalized"
    df[[new_column_name]] = scaler.fit_transform(df[[column_name]])
    
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