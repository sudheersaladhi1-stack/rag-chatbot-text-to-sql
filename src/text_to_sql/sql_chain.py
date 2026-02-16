from langchain_openai import ChatOpenAI
from .schema_loader import get_schema
from .sql_prompt import SQL_PROMPT

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

def generate_sql(question: str) -> str:
    schema = get_schema()

    if not schema:
        return "-- Error: Database is empty. Please upload a CSV/Excel file first."

    # Build schema text
    schema_text = "\n".join(
        f"Table {table}: {', '.join(cols)}"
        for table, cols in schema.items()
    )

    try:
        # ✅ Fixed: SQL_PROMPT is now a plain string, so .format() works correctly
        prompt_text = SQL_PROMPT.format(
            schema=schema_text,
            question=question
        )
        response = llm.invoke(prompt_text)

        # Clean up any accidental markdown code fences
        sql = response.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    except Exception as e:
        return f"-- Error generating SQL: {str(e)}"