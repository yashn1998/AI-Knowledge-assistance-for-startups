import pandas as pd
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import OutputParserException

# -----------------------
# Example Structured Schema
# -----------------------
class CaseAnalysis(BaseModel):
    root_cause: str = Field(description="The identified root cause of the issue")
    severity: str = Field(description="Severity level: Low, Medium, High, Critical")
    resolution_steps: str = Field(description="Steps recommended to resolve the issue")
    urgency_score: int = Field(description="Numeric urgency score from 1 (low) to 10 (critical)")


def process_dataframe_with_llm(df: pd.DataFrame, system_prompt: str) -> pd.DataFrame:
    """
    Processes a dataframe row by row using Azure OpenAI through LangChain.
    Each row must contain: case_number, problem_description, customer_symptoms.
    It will return structured JSON mapped to 4 new dataframe columns + token usage columns.
    """

    # LLM setup (adjust deployment/model names as per your Azure setup)
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o-mini",  # change this to your deployment
        model="gpt-4o-mini",
        temperature=0,
        api_version="2024-05-01-preview"
    )

    # Output parser
    parser = PydanticOutputParser(pydantic_object=CaseAnalysis)

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", 
         "Case Number: {case_number}\n"
         "Problem Description: {problem_description}\n"
         "Customer Symptoms: {customer_symptoms}\n\n"
         "Return the structured analysis in JSON.")
    ])

    # Add parser instructions
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # Columns to hold output
    df["root_cause"] = None
    df["severity"] = None
    df["resolution_steps"] = None
    df["urgency_score"] = None
    df["input_tokens"] = None
    df["output_tokens"] = None

    for idx, row in df.iterrows():
        try:
            # Build the prompt
            formatted_prompt = prompt.format_messages(
                case_number=row["case_number"],
                problem_description=row["problem_description"],
                customer_symptoms=row["customer_symptoms"]
            )

            # Count input tokens
            input_text = " ".join([m.content for m in formatted_prompt])
            input_tokens = count_tokens(input_text)

            # Call LLM
            response = llm.invoke(formatted_prompt)

            # Count output tokens
            output_tokens = count_tokens(response.content)

            # Parse structured response
            parsed: CaseAnalysis = parser.parse(response.content)

            # Assign to dataframe
            df.at[idx, "root_cause"] = parsed.root_cause
            df.at[idx, "severity"] = parsed.severity
            df.at[idx, "resolution_steps"] = parsed.resolution_steps
            df.at[idx, "urgency_score"] = parsed.urgency_score
            df.at[idx, "input_tokens"] = input_tokens
            df.at[idx, "output_tokens"] = output_tokens

        except OutputParserException as e:
            print(f"Row {idx} parsing failed: {e}")
            continue
        except Exception as e:
            print(f"Row {idx} failed: {e}")
            continue

    return df