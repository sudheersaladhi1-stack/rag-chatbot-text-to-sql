import os
from sqlalchemy import create_engine, text

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={
        "ssl": {
            "ssl_mode": "REQUIRED"
        }
    },
    pool_pre_ping=True,
    pool_recycle=3600,
)
from sqlalchemy import inspect

def get_schema():
    inspector = inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        schema[table_name] = columns
    return schema # Ensure this returns {} at minimum, never None

def run_sql(sql: str):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return result.fetchall(), result.keys()
