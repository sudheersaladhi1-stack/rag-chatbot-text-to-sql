"""Text-to-SQL chain with enhanced schema awareness."""

from langchain_openai import ChatOpenAI
from .schema_loader import get_schema
from .sql_prompt import SQL_PROMPT


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)


def format_schema_for_llm(schema: dict) -> str:
    """
    Format enriched schema into a clear, detailed description for the LLM.
    
    Args:
        schema: Dictionary with table -> {columns: [...]} structure
    
    Returns:
        Formatted string with table names, column names, types, and sample values
    """
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
            
            # Format sample values based on type
            if samples:
                if isinstance(samples[0], str):
                    samples_str = ", ".join(f"'{s}'" for s in samples[:3])
                else:
                    samples_str = ", ".join(str(s) for s in samples[:3])
                
                lines.append(
                    f"  - {col_name} ({col_type}) "
                    f"— sample values: {samples_str}"
                )
            else:
                lines.append(f"  - {col_name} ({col_type})")
    
    return "\n".join(lines)


def generate_sql(question: str) -> str:
    """
    Generate SQL query from natural language question.
    
    Args:
        question: User's natural language query
    
    Returns:
        SQL query string or error message
    """
    schema = get_schema()

    if not schema:
        return "-- Error: Database is empty. Please upload a CSV/Excel file first."

    # Format enriched schema for the LLM
    schema_text = format_schema_for_llm(schema)

    try:
        response = llm.invoke(
            SQL_PROMPT.format(
                schema=schema_text,
                question=question
            )
        )
        return response.content.strip()
    
    except Exception as e:
        return f"-- Error generating SQL: {str(e)}"