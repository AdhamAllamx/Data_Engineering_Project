from kafka import KafkaConsumer
import pandas as pd
import json
from cleaning import clean_stream_processing  
from db import save_row_to_db 

CLEANED_DATA_SET = "fintech_data_MET_P1_52_24625_clean"

def initialize_kafka_conumer():
    print("> Initializing kafka consumer.")
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        'de_ms2_52_24625',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    return consumer

def consume_kafka_stream(consumer):
    print("> Starting Kafka Consumer")
    print("> Listening for messages in 'de_ms2_52_24625'...")

    for message in consumer:
        msg_value = message.value
        print("> Message recieved form producer : ",msg_value)
        if msg_value == 'EOF':
            print("Received 'EOF' message. Stopping consumer.")
            break
        print("> Recieved new streamed row from producer ")
        new_row  = pd.DataFrame([msg_value],columns=['Customer Id','Emp Title','Emp Length','Home Ownership','Annual Inc','Annual Inc Joint','Verification Status','Zip Code','Addr State','Avg Cur Bal','Tot Cur Bal','Loan Id','Loan Status','Loan Amount','State','Funded Amount','Term','Int Rate','Grade','Issue Date','Pymnt Plan','Type','Purpose','Description'])   
        cleaned_row , lookup_table = clean_stream_processing(new_row) 
        save_row_to_db(cleaned_row, CLEANED_DATA_SET)
        print(f"> Saved cleaned streamed row to database table '{CLEANED_DATA_SET}'")
    print("> Closing Kafka Consumer")
    consumer.close()   
            
                    

