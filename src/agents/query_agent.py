# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)
import re
import string
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing import List

import openai

from src.agents.state import DeepSearchState
from langchain_core.prompts import PromptTemplate

load_dotenv()

import google.generativeai as genai

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for patent research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )

def planning_deep_research_agent(state: DeepSearchState,
                                 model: str = "gemini"):
    """
    create a search query as a text
    :param state:
    :param model:
    :return:
    """
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]

    if 'gpt' in model:
        return {"patent_search_query": create_query_by_openai(topic)}
    elif 'gemini' in model:
        return {"patent_search_query": create_query_by_gemini(topic)}


def planning_chat_deep_research_agent(state: DeepSearchState,
                                      model: str = "gemini"):
    """
    create a search query as a text
    :param state:
    :param model:
    :return:
    """
    question = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]

    if 'gemini' in model:
        return create_queries_by_gemini(question)


def create_query_by_openai(research_topic: str):
    """
    Create a search query
    Args:
        research_topic:

    Returns:

    """
    patent_query_prompt_template = f"""Your task is to construct a sophisticated patent search query. 
         Requirements:
                - Ensure the query query is self-contained and includes necessary concepts for patent search.
                - provide the query in the form of a natural-language sentence.
                - Do not include any irrelevant information to the topic/question '''{research_topic}'''.
                - Your response should encompass all key terms and concepts found in the texts for the given topic.
                - Keep it concise.
                - Do not hallucinate.

        <EXAMPLE>
            "research_topic": "cold plasma for skin treatment",
            "query": "cold plasma technology for skin treatment, focusing on its therapeutic effects for wound healing, acne, skin rejuvenation, and other dermatological applications."
        </EXAMPLE>
         <EXAMPLE>
            "research_question": "How is plasma used for nail surface treatment?",
            "query": "use of plasma in nail surface modification, fungal infection treatment, nails cleaning, nail care, and nail surface."
        </EXAMPLE>

        Return only the query text.
        """

    prompt_template = PromptTemplate(
        input_variables=["research_topic"],
        template=patent_query_prompt_template
    )
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1
    )
    output_parser = StrOutputParser()
    chain = prompt_template | llm_model | output_parser
    response = chain.invoke({"research_topic": research_topic})
    translator = str.maketrans('', '', string.punctuation)
    response = response.translate(translator)
    response = research_topic + ". " + response

    return response


def create_query_by_gemini(research_topic: str):
    """
    Create a search query
    Args:
        research_topic:

    Returns:

    """
    patent_query_prompt_template = f"""Your task is to construct a sophisticated patent search query. 
         Requirements:
                - Ensure the query query is self-contained and includes necessary concepts for patent search.
                - provide the query in the form of a natural-language sentence.
                - Do not include any irrelevant information to the topic/question '''{research_topic}'''.
                - Your response should encompass all key terms and concepts found in the texts for the given topic.
                - Keep it concise.
                - Do not hallucinate.
                
         <EXAMPLE>
            "research_topic": "cold plasma for skin treatment",
            "query": "cold plasma technology for skin treatment, focusing on its therapeutic effects for wound healing, acne, skin rejuvenation, and other dermatological applications."
        </EXAMPLE>
        <EXAMPLE>
            "research_question": "How is plasma used for nail surface treatment?",
            "query": "use of plasma in nail surface modification, fungal infection treatment, nails cleaning, nail care, and nail surface."
        </EXAMPLE>
                
        <TOPIC>
        {research_topic}
        </TOPIC>

        Return only the query text.
        """
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(patent_query_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.6,
                                          top_k=5,
                                          temperature=0.1)
                                      )
    translator = str.maketrans('', '', string.punctuation)
    response = response.text.translate(translator)
    response = research_topic + ". " + response

    return response


def create_queries_by_gemini(question: str):
    """
    Create a search query
    Args:
        question:

    Returns:

    """

    patent_query_prompt_template = f"""Your task is to construct a sophisticated patent search queries for a provided question. 
         The queries should take the form of a natural-language sentence, similar to how concepts are expressed in patent documents, 
         and it should describe the given research question in detail. 
         The response should capture both the broad technical scope and the specific features of the subject, ensuring alignment with patent-related terminology and context.
               
                - Requirements:
                - Each query should focus on one specific aspect of the provided question.
                - Each query should include all concepts in the provided question.
                - Don't produce more than 3 queries.
                - Don't generate multiple similar queries, 1 is enough.
                - Do not include any irrelevant or speculative information.
                
                Format: 
                - Format your response as a JSON object with ALL two of these exact keys:
                   - "rationale": Brief explanation of why these queries are relevant
                   - "query": A list of search queries
                
                Example:
                
                question: What are apparatus, devices, and methods that are used for plasma nail surface treatment?
                ```json
                {{
                    "rationale": "To answer this comparative growth question accurately, we need specific concepts. These queries target the precise scientific information needed: device name, methods, apparatus, technological concepts",
                    "query": ["plasma apparatus, instruments, and devices are commonly employed in plasma-based treatments for modifying or functionalizing nail surfaces, nail treatment,, including both cosmetic and biomedical applications", "methods, protocols, and treatment techniques used for applying cold plasma to nail surfaces for purposes such as improving adhesion, sterilization, or cleaning", "cold plasma, atmospheric plasma, or low-pressure plasma systems, are used for nail surfaces"],
                }}
                ```
                question: {question}"""

    llm_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.9,
        max_retries=2,
        api_key=gemini_api_key)

    result = llm_model.with_structured_output(SearchQueryList).invoke(patent_query_prompt_template)

    return {"search_queries": result.query}


def create_queries_by_openai(question: str):
    patent_query_prompt_template = f"""Your task is to construct a sophisticated patent search queries for a provided question. 
         The queries should take the form of a natural-language sentence, similar to how concepts are expressed in patent documents, 
         and it should describe the given research question in detail. 
         The response should capture both the broad technical scope and the specific features of the subject, ensuring alignment with patent-related terminology and context.

                - Requirements:
                - Each query should focus on one specific aspect of the provided question.
                - Each query should include all concepts in the provided question.
                - Don't produce more than 3 queries.
                - Don't generate multiple similar queries, 1 is enough.
                - Do not include any irrelevant or speculative information.

                Format: 
                - Format your response as a JSON object with ALL two of these exact keys:
                   - "rationale": Brief explanation of why these queries are relevant
                   - "query": A list of search queries

                Example:

                question: What are apparatus, devices, and methods that are used for plasma nail surface treatment?
                ```json
                {{
                    "rationale": "To answer this comparative growth question accurately, we need specific concepts. These queries target the precise scientific information needed: device name, methods, apparatus, technological concepts",
                    "query": ["plasma apparatus, instruments, and devices are commonly employed in plasma-based treatments for modifying or functionalizing nail surfaces, nail treatment,, including both cosmetic and biomedical applications", "methods, protocols, and treatment techniques used for applying cold plasma to nail surfaces for purposes such as improving adhesion, sterilization, or cleaning", "cold plasma, atmospheric plasma, or low-pressure plasma systems, are used for nail surfaces"],
                }}
                ```
                question: {question}"""

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=patent_query_prompt_template
    )
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1
    )
    output_parser = StrOutputParser()
    chain = prompt_template | llm_model | output_parser
    response = chain.invoke({"question": question})
    translator = str.maketrans('', '', string.punctuation)
    response = response.translate(translator)
    response = question + ". " + response

    return response




if __name__ == "__main__":
    response = create_queries_by_gemini("How can nonthermal plasma, cold plasma or non-equilibrium plasma be applied for hair loss and hair-dye and hair removal")
    print(response)
