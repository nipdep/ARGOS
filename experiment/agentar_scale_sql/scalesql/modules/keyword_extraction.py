from typing import List

from pydantic import BaseModel, Field

try:
    from langchain.output_parsers import OutputFixingParser
except ModuleNotFoundError:
    from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence

from scalesql.prompts import get_prompt

# prompt
keyword_extraction_prompt = get_prompt(
    module="nlu",
    version="value_and_skeleton_extractor",
    input_keys=["Database Schema", "Question", "Evidence"],
)


class ExtractionOutput(BaseModel):
    database_literals: List[str] = Field(
        description="The database literals extracted from the question and evidence."
    )
    question_skeleton: str = Field(
        description="the generated question skeleton"
    )


class KeywordExtractor(RunnableSequence):
    def __init__(
            self,
            llm: BaseLanguageModel
    ):
        output_parser = JsonOutputParser(pydantic_object=ExtractionOutput)
        parser = OutputFixingParser.from_llm(llm=llm, parser=output_parser)
        prompt = keyword_extraction_prompt.partial(template_output=output_parser.get_format_instructions())
        super().__init__(prompt | llm | parser)
