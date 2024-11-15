from sqlalchemy import create_engine
import pandas as pd


engine = create_engine('postgresql://root:root@pgdatabase:5432/MS2_MET_P1_52_24625')

def save_to_db(cleaned,table_name):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print(f"Writing '{table_name}' dataframe to database")
            cleaned.to_sql(table_name, con=engine, if_exists='fail')
            print(f"Done writing '{table_name}' to database")
        except ValueError as vx:
            print('Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')

def save_row_to_db(row, table_name):
    if engine.connect():
        try:
            if isinstance(row, pd.DataFrame):
                row.to_sql(table_name, con=engine, if_exists='append', index=True)
                print(f"> Successfully cleaned streamed row to table '{table_name}'")
            else:
                row = pd.DataFrame([row])  
                row.to_sql(table_name, con=engine, if_exists='append', index=True)
                print(f"> Successfully converted and saved row to table '{table_name}'")
        except Exception as ex:
            print(f"Error while saving row: {ex}")
    else:
        print("Failed to connect to Database")
