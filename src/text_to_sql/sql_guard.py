def is_safe_sql(sql: str) -> bool:
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate"]
    return sql.lower().strip().startswith("select") and not any(
        word in sql.lower() for word in forbidden
    )
