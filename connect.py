import logging

# Set up logging configuration
log = logging.getLogger()
log.setLevel('INFO')  # Set the logging level to INFO
handler = logging.StreamHandler()  # Create a handler to output logs to the console
# Define the log message format with timestamp, level, and logger name
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)  # Attach the handler to the logger

from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

# Define the Cassandra keyspace name
KEYSPACE = "mymnist"

def createKeySpace():
    """Creates a keyspace and a table in Cassandra for storing MNIST predictions."""
    # Connect to the Cassandra cluster
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect()
    
    log.info("Creating keyspace...")
    
    try:
        # Create the keyspace with a replication strategy and factor
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
            """ % KEYSPACE)
        
        log.info("Setting keyspace...")
        # Set the keyspace to be used for the session
        session.set_keyspace(KEYSPACE)
        
        log.info("Creating table...")
        # Create a table in the keyspace to store predictions with columns for date, file name, and result
        session.execute("""
            CREATE TABLE mymnist (
               date timestamp,
               file text,
               result text,
               PRIMARY KEY (date, result)
            )
            """)
    except Exception as e:
        # Log any errors encountered during the keyspace and table creation process
        log.error("Unable to create keyspace")
        log.error(e)

def insertData(date, file, result):
    """Inserts prediction data (date, file name, result) into the Cassandra table."""
    # Connect to the Cassandra cluster and keyspace
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect(KEYSPACE)
    
    log.info("Inserting data...")
    
    try:
        # Insert the provided data into the table
        session.execute(""" 
            INSERT INTO mymnist (date, file, result)
            VALUES (%s, %s, %s);
            """, (date, file, result))
    except Exception as e:
        # Log any errors encountered during the data insertion process
        log.error("Unable to insert data")
        log.error(e)
