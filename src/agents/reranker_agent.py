# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import os
import time
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from src.agents.state import DeepSearchState

load_dotenv()

import google.generativeai as genai

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
openai_model = os.getenv('OPENAI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key


def patent_reranker(topic:str, doc:str, model:str):
    """
    Apply re-ranker agent
    :param topic:
    :param doc:
    :param model:
    :return:
    """
    if 'gpt' in model:
        return rerank_by_openai(topic, doc)
    elif 'gemini' in model:
        return rerank_by_gemini(topic, doc)



def rerank_by_gemini(topic: str, doc: str):
    """
    Use Gemini model for re-ranking
    :param topic:
    :param doc:
    :return:
    """

    rerank_prompt_template = f"""
        You are an assistant whose role is to evaluate how relevant a given patent document or passage is to a specified topic.
        You will be provided with a Topic and a Patent Document/passage.\n
        Your task is to assign a relevance score from 0 to 5, and below is your grading rubric: 
             0 = not relevant at all. 
             5 = highly relevant. \n
             
        - Your relevance score should reflect whether the document addresses the entire topic.
        For example, if the topic is “cold plasma for skin treatment”, the document must relate to both “cold plasma” and “skin treatment” in order to be considered relevant.
            
        Instructions:
            - Read the topic and the patent document carefully.
            - Determine how clearly the document relates to the given topic.
            - Do not make assumptions beyond the provided text.
            - Respond only with a single score between 0 and 5.

        Topic:  '''{topic}'''\n
        Document: '''{doc}''' \n
        Score:
        """
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(rerank_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.2,
                                          top_k=2,
                                          temperature=0.0)
                                      )
    time.sleep(5)
    return response.text.strip()


def rerank_by_openai(topic: str, doc: str):
    """
    USe GPT model for re-ranking
    :param topic:
    :param doc:
    :return:
    """

    rerank_prompt_template = f"""
        You are an assistant whose role is to evaluate how relevant a given patent document is to a specified topic.
        You will be provided with a Topic and a Patent Document.\n
        Your task is to assign a relevance score from 0 to 5, and below is your grading rubric:  
             0 = not relevant at all. 
             5 = highly relevant. \n

        - Your relevance score should reflect whether the document addresses the entire topic.
        For example, if the topic is “cold plasma for skin treatment”, the document must relate to both “cold plasma” and “skin treatment” in order to be considered relevant.

        Instructions:
            - Read the topic and the patent document carefully.
            - Determine how clearly the document relates to the given topic.
            - Do not make assumptions beyond the provided text.
            - Respond only with a single score between 0 and 5.

        Topic:  '''{topic}'''\n
        Document: '''{doc}''' \n
        Score:
        """
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1,
    )

    prompt = PromptTemplate(
        input_variables=["topic", "doc"],
        template=rerank_prompt_template,
    )
    # Create chain
    chain = prompt | llm_model | StrOutputParser()
    score = chain.invoke({"topic": topic, "document": doc})

    return score


if __name__ == "__main__":
    topic = "Cold Plasma for Hair Loss, Hair-Dye and Hair Removal"
    patent_text = "The invention is in the technical field of hair removal equipment, specifically using helium gas. The objective is to provide an improved epilation device that overcomes the limitations of existing methods like laser and photoepilation, which are ineffective on certain hair types or hair growth stages, by creating an athermic plasma plume of helium gas. This equipment is designed for permanent hair removal, regardless of hair color or growth stage. The core of the invention is an epilation equipment comprising an arc flash generator, an applicator handle, and a mechanical valve, configured to modulate helium gas flow to generate an athermic plasma plume via the arc generator."
    score = rerank_by_openai(topic, patent_text)
    print(score)
    if int(score) > 2:
        print(score)
