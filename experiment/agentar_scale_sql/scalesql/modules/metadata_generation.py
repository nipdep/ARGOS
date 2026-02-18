from typing import Type, Union, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
try:
    from langchain.output_parsers import OutputFixingParser
except ModuleNotFoundError:
    from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from scalesql.prompts import get_prompt

table_metadata_prompt = get_prompt(
    module="database_understanding",
    version="table_metadata",
    input_keys=[
        "table_name",
        "table_description",
        "table"
    ]
)


class DefaultTableMetadataOutput(BaseModel):
    """The default output schema of the table metadata generator."""
    column_annotations: Dict[str, str] = Field(
        description="The annotations of the columns, the key is the input column name, the value is the annotation.")


class TableMetadata(RunnableSequence):
    def __init__(
            self,
            llm: BaseLanguageModel,
            prompt: ChatPromptTemplate = table_metadata_prompt,
            output_schema: Union[BaseModel, Type[BaseModel]] = DefaultTableMetadataOutput,
    ):
        """
        Initialize the table metadata generator.
        Args:
            llm: BaseLanguageModel
            prompt: the prompt template
            output_schema: the output schema
        """
        output_parser = JsonOutputParser(pydantic_object=output_schema)
        output_parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)
        prompt = prompt.partial(
            template_output=output_parser.get_format_instructions()
        )
        super().__init__(
            prompt | llm | output_parser
        )
