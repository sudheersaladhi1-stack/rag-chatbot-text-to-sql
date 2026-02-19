"""SQL safety validator - allows SELECT and CTE queries only."""

def is_safe_sql(sql: str) -> bool:
    """
    Check if SQL query is safe to execute.
    
    Allows:
        - SELECT queries
        - CTEs (WITH ... AS ... SELECT)
    
    Blocks:
        - INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE
        - Any query that doesn't start with SELECT or WITH
    
    Args:
        sql: SQL query string to validate
    
    Returns:
        True if safe, False otherwise
    """
    if not sql or not sql.strip():
        return False
    
    sql_lower = sql.lower().strip()
    
    # Allow queries starting with SELECT or WITH (for CTEs)
    valid_starts = sql_lower.startswith("select") or sql_lower.startswith("with")
    
    # Block dangerous operations
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", 
                 "create", "grant", "revoke", "exec", "execute"]
    
    has_forbidden = any(word in sql_lower for word in forbidden)
    
    return valid_starts and not has_forbidden