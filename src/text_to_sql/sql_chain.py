from langchain_openai import ChatOpenAI
from .schema_loader import get_schema
from .sql_prompt import SQL_PROMPT

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

def generate_sql(question: str) -> str:
    schema = get_schema()

    schema_text = "\n".join(
        f"{table}: {', '.join(cols)}"
        for table, cols in schema.items()
    )

    response = llm.invoke(
        SQL_PROMPT.format(
            schema=schema_text,
            question=question
        )
    )

    return response.strip()
