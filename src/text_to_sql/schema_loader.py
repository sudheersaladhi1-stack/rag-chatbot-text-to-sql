"""Enhanced schema loader with data types and sample values for better SQL generation."""

from sqlalchemy import inspect, text
from .db import engine


def get_schema():
    """
    Return enriched schema with column names, types, and sample values.
    
    Returns:
        dict: {
            "table_name": {
                "columns": [
                    {"name": "col1", "type": "INTEGER", "sample_values": [1, 2, 3]},
                    {"name": "col2", "type": "VARCHAR", "sample_values": ["a", "b", "c"]},
                ]
            }
        }
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            return {}

        schema_dict = {}
        
        for table in tables:
            cols_info = inspector.get_columns(table)
            enriched_columns = []
            
            for col in cols_info:
                col_name = col["name"]
                col_type = str(col["type"])
                
                # Fetch sample values (up to 5 distinct values)
                sample_values = []
                try:
                    with engine.connect() as conn:
                        query = text(
                            f"SELECT DISTINCT `{col_name}` FROM `{table}` "
                            f"WHERE `{col_name}` IS NOT NULL "
                            f"LIMIT 5"
                        )
                        result = conn.execute(query)
                        sample_values = [row[0] for row in result]
                except Exception:
                    sample_values = []  # Skip if query fails
                
                enriched_columns.append({
                    "name": col_name,
                    "type": col_type,
                    "sample_values": sample_values,
                })
            
            schema_dict[table] = {"columns": enriched_columns}
        
        return schema_dict
    
    except Exception as e:
        print(f"Error fetching schema: {e}")
        return {}


def get_schema_legacy():
    """
    Legacy schema format for backward compatibility.
    Returns: {"table_name": ["col1", "col2", ...]}
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            return {}
        
        schema_dict = {}
        for table in tables:
            cols = inspector.get_columns(table)
            schema_dict[table] = [c["name"] for c in cols]
        
        return schema_dict
    except Exception as e:
        print(f"Error fetching schema: {e}")
        return {}