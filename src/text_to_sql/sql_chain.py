"""Text-to-SQL chain with CTE-based queries and insight extraction."""

import re
from langchain_openai import ChatOpenAI
from .schema_loader import get_schema
from .sql_prompt import SQL_PROMPT


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    streaming=True
)


def format_schema_for_llm(schema: dict) -> str:
    """Format enriched schema for the LLM."""
    if not schema:
        return "No tables available."
    
    lines = []
    for table_name, table_info in schema.items():
        lines.append(f"\nTable: {table_name}")
        lines.append("Columns:")
        
        for col in table_info["columns"]:
            col_name = col["name"]
            col_type = col["type"]
            samples = col["sample_values"]
            
            if samples:
                if isinstance(samples[0], str):
                    samples_str = ", ".join(f"'{s}'" for s in samples[:3])
                else:
                    samples_str = ", ".join(str(s) for s in samples[:3])
                
                lines.append(
                    f"  - {col_name} ({col_type}) — sample values: {samples_str}"
                )
            else:
                lines.append(f"  - {col_name} ({col_type})")
    
    return "\n".join(lines)


def extract_thinking_and_sql(response: str) -> tuple[str, str]:
    """
    Extract thinking (insights only) and SQL separately.
    Remove any SQL code from the thinking section.
    
    Returns:
        (thinking_text, sql_query)
    """
    # Extract thinking section
    thinking_match = re.search(
        r"💭\s*\*\*Thinking:\*\*(.*?)```sql",
        response,
        re.DOTALL | re.IGNORECASE
    )
    
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    else:
        # Fallback: everything before first ```sql
        parts = response.split("```sql")
        thinking = parts[0].replace("💭", "").replace("**Thinking:**", "").strip()
    
    # Remove any SQL-like patterns from thinking
    # (table names, WHERE clauses, SELECT statements)
    sql_patterns = [
        r'\bSELECT\b.*',
        r'\bFROM\b.*',
        r'\bWHERE\b.*',
        r'\bGROUP BY\b.*',
        r'\bORDER BY\b.*',
        r'\bWITH\b\s+\w+\s+AS\b.*',
    ]
    
    for pattern in sql_patterns:
        thinking = re.sub(pattern, '', thinking, flags=re.IGNORECASE | re.DOTALL)
    
    # Extract SQL query
    sql_match = re.search(
        r"```sql\s*(.*?)\s*```",
        response,
        re.DOTALL | re.IGNORECASE
    )
    
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # Fallback: check if entire response is SQL
        if response.strip().upper().startswith(("SELECT", "WITH")):
            sql = response.strip()
        else:
            sql = ""
    
    return thinking, sql


def generate_sql(question: str, stream_callback=None) -> dict:
    """
    Generate SQL query with business insights.
    
    Args:
        question: User's natural language query
        stream_callback: Optional callback(chunk: str) for streaming
    
    Returns:
        dict with:
            - 'thinking': Business insights only (no SQL)
            - 'sql': SQL query string
            - 'full_response': Complete LLM output
    """
    schema = get_schema()

    if not schema:
        return {
            "thinking": "Database is empty. Please upload data first.",
            "sql": "",
            "full_response": "-- Error: Database is empty."
        }

    schema_text = format_schema_for_llm(schema)

    try:
        if stream_callback:
            # Streaming mode
            full_response = ""
            for chunk in llm.stream(
                SQL_PROMPT.format(
                    schema=schema_text,
                    question=question
                )
            ):
                token = chunk.content
                full_response += token
                stream_callback(token)
        else:
            # Non-streaming mode
            response = llm.invoke(
                SQL_PROMPT.format(
                    schema=schema_text,
                    question=question
                )
            )
            full_response = response.content
        
        # Extract thinking and SQL separately
        thinking, sql = extract_thinking_and_sql(full_response)
        
        return {
            "thinking": thinking,
            "sql": sql,
            "full_response": full_response.strip()
        }
    
    except Exception as e:
        return {
            "thinking": f"Error: {str(e)}",
            "sql": "",
            "full_response": f"-- Error generating SQL: {str(e)}"
        }