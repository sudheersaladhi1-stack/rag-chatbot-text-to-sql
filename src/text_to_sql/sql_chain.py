"""Text-to-SQL chain with insights generation and SQL-free thinking."""

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
    Extract thinking (NO SQL) and SQL separately.
    Aggressively remove ALL SQL-like content from thinking.
    
    Returns:
        (thinking_text, sql_query)
    """
    # Extract SQL query first
    sql_match = re.search(
        r"```sql\s*(.*?)\s*```",
        response,
        re.DOTALL | re.IGNORECASE
    )
    
    if sql_match:
        sql = sql_match.group(1).strip()
        # Remove everything from first ```sql onwards to get thinking
        thinking = response.split("```sql")[0]
    else:
        # No SQL found
        if response.strip().upper().startswith(("SELECT", "WITH")):
            sql = response.strip()
            thinking = ""
        else:
            sql = ""
            thinking = response
    
    # Clean thinking section
    thinking = thinking.replace("💭", "").replace("**Thinking:**", "").strip()
    
    # AGGRESSIVELY remove ALL SQL-like patterns from thinking
    # Remove entire lines containing SQL keywords
    sql_keywords = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 
        'WITH', 'AS (', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'ON ', '= ', 'SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(',
        'DISTINCT', 'HAVING', 'LIMIT', 'OFFSET', 'UNION', 'CASE WHEN'
    ]
    
    lines = thinking.split('\n')
    cleaned_lines = []
    for line in lines:
        line_upper = line.upper()
        has_sql = any(keyword in line_upper for keyword in sql_keywords)
        if not has_sql and line.strip():
            cleaned_lines.append(line)
    
    thinking = '\n'.join(cleaned_lines).strip()
    
    return thinking, sql


def generate_insights_from_results(df, question: str) -> str:
    """
    Generate business insights from query results.
    
    Args:
        df: pandas DataFrame with results
        question: original user question
    
    Returns:
        Insight text (e.g., "Highest: Product A ($50K), Lowest: Product Z ($5K)")
    """
    if df.empty or len(df) == 0:
        return ""
    
    insights = []
    
    # Check if there's a numeric column (sales, revenue, amount, etc.)
    numeric_cols = df.select_dtypes(include=['number']).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0 and len(text_cols) > 0:
        # Likely a ranking or comparison query
        metric_col = numeric_cols[0]
        label_col = text_cols[0]
        
        # Get top and bottom
        if len(df) >= 2:
            top_row = df.iloc[0]
            bottom_row = df.iloc[-1]
            
            top_label = top_row[label_col]
            top_value = top_row[metric_col]
            bottom_label = bottom_row[label_col]
            bottom_value = bottom_row[metric_col]
            
            insights.append(
                f"**Highest:** {top_label} ({top_value:,.0f})"
            )
            insights.append(
                f"**Lowest:** {bottom_label} ({bottom_value:,.0f})"
            )
            
            # Calculate spread
            if bottom_value > 0:
                spread = ((top_value - bottom_value) / bottom_value) * 100
                insights.append(
                    f"**Spread:** {spread:.1f}% difference between top and bottom"
                )
    
    return " | ".join(insights) if insights else ""


def generate_sql(question: str, stream_callback=None) -> dict:
    """
    Generate SQL query with business insights.
    
    Args:
        question: User's natural language query
        stream_callback: Optional callback(chunk: str) for streaming
    
    Returns:
        dict with:
            - 'thinking': Business insights only (NO SQL)
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
        
        # Extract thinking (SQL-free) and SQL separately
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