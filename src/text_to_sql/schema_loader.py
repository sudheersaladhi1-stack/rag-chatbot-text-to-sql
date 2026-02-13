from sqlalchemy import inspect
from .db import engine

def get_schema():
    inspector = inspect(engine)
    schema = {}

    for table in inspector.get_table_names():
        schema[table] = [
            col["name"] for col in inspector.get_columns(table)
        ]

    return schema
