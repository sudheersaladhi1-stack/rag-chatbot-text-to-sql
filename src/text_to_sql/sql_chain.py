from langchain_openai import ChatOpenAI
from .schema_loader import get_schema
from .sql_prompt import SQL_PROMPT

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

def generate_sql(question: str) -> str:
    schema = get_schema()

    # Guard against empty schema
    if not schema:
        return "-- Error: Database is empty. Please upload a CSV/Excel file first."

    # Now .items() works because schema is a dictionary!
    schema_text = "\n".join(
        f"Table {table}: {', '.join(cols)}"
        for table, cols in schema.items()
    )

    try:
        response = llm.invoke(
            SQL_PROMPT.format(
                schema=schema_text,
                question=question
            )
        )
        
        # Extract content from the AIMessage object
        return response.content.strip()
    except Exception as e:
        return f"-- Error generating SQL: {str(e)}"