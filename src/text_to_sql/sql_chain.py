"""Text-to-SQL chain with aggressive SQL removal from thinking."""

import re
from datetime import date
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
    
    Ultra-aggressive: Remove ENTIRE thinking if it contains ANY SQL.
    
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
        sql_start_idx = response.upper().find(sql[:30].upper() if len(sql) >= 30 else sql.upper())
        if sql_start_idx > 0:
            thinking_raw = response[:sql_start_idx]
        else:
            thinking_raw = ""
    else:
        thinking_raw = response
    
    # Clean thinking
    thinking_raw = thinking_raw.replace("💭", "").replace("**Thinking:**", "").replace("Thinking:", "").strip()
    
    # ULTRA-AGGRESSIVE: Remove ALL lines containing SQL patterns
    # Expanded list of SQL indicators
    sql_indicators = [
        # SQL keywords
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT',
        'WITH', 'AS (', 'AS(', 'CTE', 'UNION', 'INTERSECT', 'EXCEPT',
        # SQL functions
        'SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(', 'ROUND(', 'FLOOR(', 'CEIL(',
        'CONCAT(', 'SUBSTRING(', 'UPPER(', 'LOWER(', 'TRIM(',
        'DATE(', 'YEAR(', 'MONTH(', 'DAY(', 'NOW(', 'CURRENT_',
        # Table/column patterns
        '_ID', '_NAME', '_AMOUNT', '_DATE', '_FACT', '_DIM', '_TABLE',
        'SALES_', 'PRODUCT_', 'STORE_', 'CUSTOMER_', 'ORDER_',
        # Technical terms
        'ALIAS', 'FOREIGN KEY', 'PRIMARY KEY', 'INDEX', 'CONSTRAINT',
        'PARTITION', 'SUBQUERY', 'AGGREGATE', 'WINDOW FUNCTION',
    ]
    
    lines = thinking_raw.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 10:  # Skip empty or very short lines
            continue
            
        line_upper = line_stripped.upper()
        
        # Check if line contains ANY SQL indicator
        has_sql = any(indicator in line_upper for indicator in sql_indicators)
        
        # Additional heuristics for SQL detection
        if '(' in line and ')' in line and '=' in line:  # SQL expressions
            has_sql = True
        if line_stripped.count(',') >= 2:  # Column lists
            has_sql = True
        if any(char in line for char in [';', '`']):  # SQL punctuation
            has_sql = True
        if line_upper.startswith(('TABLE', 'COLUMN', 'DATABASE', 'SCHEMA')):
            has_sql = True
            
        if not has_sql:
            cleaned_lines.append(line_stripped)
    
    thinking = ' '.join(cleaned_lines).strip()
    
    # Final cleanup: If thinking still looks like SQL, replace entirely
    if not thinking or len(thinking) < 15:
        thinking = "Analyzing data to extract business insights."
    
    # Extra safety: If ANY SQL keyword remains, clear it
    final_check = thinking.upper()
    if any(kw in final_check for kw in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP', 'ORDER']):
        thinking = "Generating insights from the data analysis."
    
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
            today_str = date.today().strftime("%Y-%m-%d")
            for chunk in llm.stream(SQL_PROMPT.format(schema=schema_text, question=question, today=today_str)):
                token = chunk.content
                full_response += token
                stream_callback(token)
        else:
            today_str = date.today().strftime("%Y-%m-%d")
            response = llm.invoke(SQL_PROMPT.format(schema=schema_text, question=question, today=today_str))
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