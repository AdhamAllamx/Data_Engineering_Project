from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from functions import extract_clean, transform, load_to_db
from fintech_dashboard import run_dashboard


# Define the DAG
default_args = {
    "owner": "fintech_allam_52_24625",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'fintech_etl_pipeline',
    default_args=default_args,
    description='fintech etl pipeline',
)

with DAG(
    dag_id = 'fintech_etl_pipeline',
    schedule_interval = '@once', # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args = default_args,
    tags = ['fintech-pipeline'],
)as dag:
    # Define the tasks
    extract_clean_task = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_data_2_52_24625.csv'
        }
    )

    transform_task = PythonOperator(
        task_id = 'transform',
        python_callable = transform,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_clean.csv'
        }
    )

    load_to_db_task = PythonOperator(
        task_id = 'load_to_db',
        python_callable = load_to_db,
        op_kwargs = {
            'filename': '/opt/airflow/data/fintech_transformed.csv'
        }
    )

    run_dashboard_task = BashOperator(
        task_id="run_dashboard",
        bash_command=(
            "streamlit run /opt/airflow/dags/fintech_dashboard.py "
            "--server.port 8501 "
            "--server.headless true"
        )
    )


    # Define the task dependencies
    extract_clean_task >> transform_task >> load_to_db_task >> run_dashboard_task





