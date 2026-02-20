"""Text-to-SQL chain with aggressive SQL removal from thinking."""

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
    Extract thinking (completely SQL-free) and SQL separately.
    
    Strategy:
    1. Find SQL query in ```sql blocks
    2. Extract everything before first SQL keyword as thinking
    3. Aggressively remove ANY line containing SQL patterns
    
    Returns:
        (thinking_text, sql_query)
    """
    # Extract SQL first
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # Try to find SQL without code block markers
        if response.strip().upper().startswith(("SELECT", "WITH")):
            sql = response.strip()
        else:
            sql = ""
    
    # Extract thinking - everything before SQL appears
    if "```sql" in response:
        thinking_raw = response.split("```sql")[0]
    elif sql:
        # SQL found but no markers - split before it
        sql_start = response.upper().find(sql[:20].upper())
        if sql_start > 0:
            thinking_raw = response[:sql_start]
        else:
            thinking_raw = ""
    else:
        thinking_raw = response
    
    # Clean thinking
    thinking_raw = thinking_raw.replace("💭", "").replace("**Thinking:**", "").strip()
    
    # AGGRESSIVELY remove ALL SQL-like lines
    sql_patterns = [
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET',
        'WITH', 'AS (', 'AS(', 'CTE', 
        'SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(', 'ROUND(',
        'DISTINCT', 'UNION', 'INTERSECT', 'EXCEPT',
        'ON ', '= ', 'AND ', 'OR ',
        '_id', '_name', '_amount', '_date', '_fact', '_dim',
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER'
    ]
    
    lines = thinking_raw.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_upper = line_stripped.upper()
        
        # Check if line contains ANY SQL pattern
        has_sql = any(pattern in line_upper for pattern in sql_patterns)
        
        # Also check for common SQL punctuation patterns
        if '(' in line and ')' in line and '=' in line:
            has_sql = True
        if line_stripped.count(',') > 2:  # Multiple commas suggest column list
            has_sql = True
        if line_upper.startswith(('TABLE', 'COLUMN', 'INDEX', 'DATABASE')):
            has_sql = True
            
        if not has_sql:
            cleaned_lines.append(line_stripped)
    
    thinking = ' '.join(cleaned_lines).strip()
    
    # If thinking is too short or empty, provide generic fallback
    if len(thinking) < 20:
        thinking = "Analyzing data to provide business insights."
    
    return thinking, sql


def generate_insights_from_results(df, question: str) -> str:
    """Generate business insights from query results."""
    if df.empty or len(df) == 0:
        return ""
    
    insights = []
    numeric_cols = df.select_dtypes(include=['number']).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0 and len(text_cols) > 0:
        metric_col = numeric_cols[0]
        label_col = text_cols[0]
        
        if len(df) >= 2:
            top_row = df.iloc[0]
            bottom_row = df.iloc[-1]
            
            top_label = top_row[label_col]
            top_value = top_row[metric_col]
            bottom_label = bottom_row[label_col]
            bottom_value = bottom_row[metric_col]
            
            insights.append(f"**Highest:** {top_label} ({top_value:,.0f})")
            insights.append(f"**Lowest:** {bottom_label} ({bottom_value:,.0f})")
            
            if bottom_value > 0:
                spread = ((top_value - bottom_value) / bottom_value) * 100
                insights.append(f"**Spread:** {spread:.1f}% difference")
    
    return " | ".join(insights) if insights else ""


def generate_sql(question: str, stream_callback=None) -> dict:
    """
    Generate SQL query with business insights.
    
    Returns:
        dict with 'thinking' (no SQL), 'sql', 'full_response'
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
            full_response = ""
            for chunk in llm.stream(SQL_PROMPT.format(schema=schema_text, question=question)):
                token = chunk.content
                full_response += token
                stream_callback(token)
        else:
            response = llm.invoke(SQL_PROMPT.format(schema=schema_text, question=question))
            full_response = response.content
        
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