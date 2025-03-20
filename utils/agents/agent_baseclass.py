from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional



class QueryClassfication(BaseModel):
    """
    This tool is used to classify the the user query. 
    """
    is_python: bool = Field(
        description="Whether the user is asking a python related question or not."
    )
    is_sql: bool = Field(
        description="Whether the user is asking a sql related question or not."
    )
    is_other: bool = Field(
        description="Whether the user is asking a question related to other languages apart from python and sql."
    )



class PythonCode(BaseModel):
    """
    This tool is used to generate python code.
    """
    code: str = Field(
        description="The generated python code."
    )



class SqlCode(BaseModel):
    """
    This tool is used to generate sql code.
    """
    code: str = Field(
        description="The generated sql code."
    )


class Summarization(BaseModel):
    """
    This tool is used to summarize the given text.
    """
    summary: str = Field(
        description="The summarized text."
    )