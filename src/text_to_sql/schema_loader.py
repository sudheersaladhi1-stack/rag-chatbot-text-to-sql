from sqlalchemy import inspect
from .db import engine

def get_schema():
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return {} # Always return a dictionary, even if empty

        schema_dict = {}
        for table in tables:
            cols = inspector.get_columns(table)
            # Map table name to a list of column names
            schema_dict[table] = [c["name"] for c in cols]

        return schema_dict
    except Exception as e:
        print(f"Error fetching schema: {e}")
        return {}