"""LLM-backed natural-language → canonical rule parsing.

Extracted from the former ``text2rules-v2.py`` batch script so the UI can
parse a single rule at authoring time and show the user its encoded form
immediately. The pydantic ``Rule`` schema here is the canonical structured
shape shared with every format importer (see ``rules_parsing/canonical.py``).
"""
import os
from typing import Union, Literal, Optional

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain.globals import set_llm_cache
set_llm_cache(None)


class Unary_predicate(BaseModel):
    operator: Optional[Literal["NOT"]] = None
    name: str


class Binary_predicate(BaseModel):
    operator: Literal["AND", "OR"]
    arg1: Unary_predicate
    arg2: Unary_predicate


class Predicate(BaseModel):
    operator: Literal["AND", "OR"]
    arg1: Union[Unary_predicate, Binary_predicate]
    arg2: Union[Unary_predicate, Binary_predicate]


class Rule(BaseModel):
    if_part: Predicate
    then_part: Unary_predicate


Rule.model_rebuild()

_output_parser = PydanticOutputParser(pydantic_object=Rule)

_PROMPT = PromptTemplate(
    input_variables=["sentence", "predicates"],
    template="""
    You are a logical proposition parser.
    Convert the following natural language sentence into a logical expression
    using only the operators AND, OR, and NOT.

    Only use predicates from this list: {predicates}

    {format_instructions}

    Sentence: {sentence}
    """,
    partial_variables={"format_instructions": _output_parser.get_format_instructions()},
)


def parse_rule(sentence: str, predicate_names: list, seed: int = 42) -> dict:
    """Parse one natural-language rule into the canonical rule dict.

    Raises whatever the LLM chain raises (missing OPENAI_API_KEY, quota,
    unparseable output) — callers surface the error to the user and drop
    the rule.
    """
    llm = ChatOpenAI(model="gpt-5", temperature=0, seed=seed,
                     api_key=os.getenv("OPENAI_API_KEY"))
    chain = _PROMPT | llm | _output_parser
    result = chain.invoke({
        "sentence": sentence,
        "predicates": ", ".join(predicate_names),
    })
    return result.model_dump()
