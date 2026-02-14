from sqlalchemy import inspect
from .db import engine

def get_schema():
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return {} # Return empty dict instead of a string

        schema = {}
        for table in tables:
            cols = inspector.get_columns(table)
            # Store columns as a list for easier processing later
            schema[table] = [c["name"] for c in cols]

        return schema
    except Exception as e:
        print(f"Error fetching schema: {e}")
        return {}