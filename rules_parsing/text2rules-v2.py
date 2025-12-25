from typing import List, Union, Literal, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
import json
import argparse
import os
from ltn_parser import LTNParser
from lambda_parser import PythonLambdaParser 


from langchain.globals import set_llm_cache
set_llm_cache(None)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--raw_rules_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

raw_rules_path = args.raw_rules_path
output_path = args.output_path

ltn_parser = LTNParser()
python_rules_parser = PythonLambdaParser()


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

llm = ChatOpenAI(model="gpt-5", temperature=0, seed=args.seed, api_key=os.getenv("OPENAI_API_KEY"))

parser = PydanticOutputParser(pydantic_object=Rule)

def llm_parse(predicates, sentence):

    prompt = PromptTemplate(
        input_variables=["sentence", "predicates"],
        template="""
        You are a logical proposition parser.
        Convert the following natural language sentence into a logical expression
        using only the operators AND, OR, and NOT.

        Only use predicates from this list: {predicates}

        {format_instructions}

        Sentence: {sentence}
        """,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    parser_chain = prompt | llm  | parser
    result = parser_chain.invoke({"sentence": sentence, "predicates": ", ".join(predicates)})
    return result

def parse_sentences(predicates, sentences):
    results = []
    for sentence in sentences:
        parsed = llm_parse(predicates, sentence)
        ltn_rule = ltn_parser.parse_rule(json.loads(parsed.model_dump_json()))
        rules_only_rule = python_rules_parser.parse_rule(json.loads(parsed.model_dump_json()))
        results.append({"text": sentence, "json": parsed, "ltn": ltn_rule, "rules_only": rules_only_rule})
    return results

def parse_and_write_to_files(predicates, sentences):
    rules = parse_sentences(predicates, sentences)

    ltn_rules = [f"Forall(x, {d["ltn"]})" for d in rules]
    ltn_rules_str = "" + ", ".join(ltn_rules) + ""

    ltn_code = f"parsed_rules = lambda x, y: [{ltn_rules_str}]"

    rules_only_rules = [d["rules_only"] for d in rules]
    rules_only_rules_str = ", ".join(rules_only_rules)

    rules_only_code = f"parsed_rules_python = lambda x, y: [{rules_only_rules_str}]"

    with open(f"{output_path}/ltn_rules.txt", "w") as text_file:
        text_file.write(ltn_code)

    with open(f"{output_path}/rules_only_rules.txt", "w") as text_file:
        text_file.write(rules_only_code)

with open(raw_rules_path, 'r') as file:
    sentences = file.read().split("\n")

with open(f"{output_path}/predicate_names.txt", 'r') as file:
    predicates = file.read().split("\n")

parse_and_write_to_files(predicates, sentences)