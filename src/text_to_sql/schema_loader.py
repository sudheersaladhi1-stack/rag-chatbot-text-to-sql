from sqlalchemy import inspect
from .db import engine

def get_schema():
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if not tables:
        return "No tables available in the database."

    schema = []
    for table in tables:
        cols = inspector.get_columns(table)
        col_names = ", ".join(c["name"] for c in cols)
        schema.append(f"Table {table}: {col_names}")

    return "\n".join(schema)
