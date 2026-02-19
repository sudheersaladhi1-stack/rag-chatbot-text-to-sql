"""Text-to-SQL chain with multi-step query execution support."""

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
    """Format enriched schema into a clear description for the LLM."""
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


def extract_sql_queries(response: str) -> list[str]:
    """
    Extract SQL queries from LLM response.
    Handles both single and multi-step queries.
    
    Returns:
        List of SQL query strings (one or more)
    """
    # Find all SQL code blocks
    sql_pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Clean up each query
        queries = [q.strip() for q in matches if q.strip()]
        return queries
    
    # Fallback: entire response might be SQL
    response_clean = response.strip()
    if response_clean.upper().startswith("SELECT"):
        return [response_clean]
    
    return []


def generate_sql(question: str, stream_callback=None) -> dict:
    """
    Generate SQL query/queries from natural language question.
    
    Args:
        question: User's natural language query
        stream_callback: Optional callback function(chunk: str) for streaming
    
    Returns:
        dict with:
            - 'full_response': Complete LLM response with thinking
            - 'queries': List of SQL queries to execute in order
            - 'is_multi_step': Boolean indicating if multiple queries
    """
    schema = get_schema()

    if not schema:
        return {
            "full_response": "-- Error: Database is empty. Please upload a CSV/Excel file first.",
            "queries": [],
            "is_multi_step": False
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
        
        # Extract SQL queries
        queries = extract_sql_queries(full_response)
        
        return {
            "full_response": full_response.strip(),
            "queries": queries,
            "is_multi_step": len(queries) > 1
        }
    
    except Exception as e:
        return {
            "full_response": f"-- Error generating SQL: {str(e)}",
            "queries": [],
            "is_multi_step": False
        }