from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.agents.state import DeepSearchState
import google.generativeai as genai

import os
from dotenv import load_dotenv
import openai

load_dotenv()
gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


def patent_deep_review(state: DeepSearchState):
    """
    LangGraph node that detects knowledge gaps and suggests follow-up queries
        - Reviews the current summary to pinpoint areas needing further investigation.
        - Generates potential follow-up queries and outputs them in a structured JSON format.
    Args:
        state:

    Returns:

    """
    if 'gpt' in state.llm:
        return patent_deep_review_by_openai(state)
    elif 'gemini' in state.llm:
        return patent_deep_review_by_gemini(state)


def patent_deep_review_by_gemini(state: DeepSearchState):
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    patent_review_prompt = f"""
                You are an expert research assistant tasked with writing a comprehensive technical scientific report for {topic} based on a provided collection of patent documents. 
                Each document includes a patent number and the associated summary. 
                The report should be detailed, formal, and structured according to the following guidelines:\n

            🔹 Report Structure:\n
                - Abstract (Summary):\n
                    Provide a concise summary of the inventions covered the topic {topic} in the provided patent texts.\n

                - Explain the background context and existing technologies or challenges the inventions address related.\n

                - Technical Fields of Invention:\n
                    Provide a detailed list of technological areas and fields of invention that related to the topic {topic}, provide them with citations.'\n

              - Inventions related to the topic/question  '''{topic}'''
                    - show with details a list of most related inventions with novelty and objectives.\n
                    - Highlight unique components, devices, apparatus, methods, or systems.\n
                    - Identify the technical problems solved and how the invention provides an improvement.\n

              - Applicability and Uses:\n
                 Discuss with details practical applications and uses of the most inventions related to the topic {topic}.\n

              - Conclusion:\n
                    Summarize the overall inventions of the patents.\n  

            🔹 Formatting and Style:
                Use formal, technical language appropriate for a research or patent analyst audience.\n      
                Reference each patent by its number (e.g., "as described in US1234567").\n
                Group similar or related inventions where appropriate to avoid redundancy.\n
                Where useful, include tables, bullet points, or diagrams (optional, based on capabilities).
        - Requirements:
            - All information in the report should relate to {topic} \n
            - Don't provide information outside the topic '''{topic}'''\n
            - Ensure that all generated content is specifically focused on the complete topic — for example, ‘cold plasma for wound healing’ — rather than discussing ‘cold plasma’ or ‘wound healing’ in isolation. The report information should clearly address how cold plasma is used within the context of wound healing, covering aspects such as underlying mechanisms, therapeutic benefits, practical applications, and supporting research.
            - Use patent numbers as citations when discussing specific inventions like (US 20180307744). This citation is provided along with the contexts, and don't provide citations outside the provided contexts.\n
            - Do not hallucinate.\n
            - Do not include any irrelevant information. \n

            PATENT SUMMARIES: '''{state.patent_research_results}''' \n
            at the end of your report, provide a list of citations that only used in the report.
            """
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(patent_review_prompt,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.6,
                                          top_k=5,
                                          temperature=0.2)
                                      )

    state.patent_running_summary = f"{response.text}\n ## Sources: \n{state.patent_sources_gathered}"

    return {"patent_running_summary": state.patent_running_summary}


def patent_deep_review_by_openai(state: DeepSearchState):
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    patent_review_prompt = f"""
                    You are an expert research assistant tasked with writing a comprehensive technical scientific report for {topic} based on a provided collection of patent documents. 
                    Each document includes a patent number and the associated summary. 
                    The report should be detailed, formal, and structured according to the following guidelines:\n

                🔹 Report Structure:\n
                    - Abstract (Summary):\n
                        Provide a concise summary of the inventions covered the topic {topic} in the provided patent texts.\n

                    - Explain the background context and existing technologies or challenges the inventions address related.\n

                    - Technical Fields of Invention:\n
                        Provide a detailed list of technological areas and fields of invention that related to the topic {topic}, provide them with citations.'\n

                  - Inventions related to the topic/question  '''{topic}'''
                        - show with details a list of most related inventions with novelty and objectives.\n
                        - Highlight unique components, devices, apparatus, methods, or systems.\n
                        - Identify the technical problems solved and how the invention provides an improvement.\n

                  - Applicability and Uses:\n
                     Discuss with details practical applications and uses of the most inventions related to the topic {topic}.\n

                  - Conclusion:\n
                        Summarize the overall inventions of the patents.\n  

                🔹 Formatting and Style:
                    Use formal, technical language appropriate for a research or patent analyst audience.\n      
                    Reference each patent by its number (e.g., "as described in US1234567").\n
                    Group similar or related inventions where appropriate to avoid redundancy.\n
                    Where useful, include tables, bullet points, or diagrams (optional, based on capabilities).
            - Requirements:
                - All information in the report should relate to {topic} \n
                - Don't provide information outside the topic '''{topic}'''\n
                - Ensure that all generated content is specifically focused on the complete topic — for example, ‘cold plasma for wound healing’ — rather than discussing ‘cold plasma’ or ‘wound healing’ in isolation. The report information should clearly address how cold plasma is used within the context of wound healing, covering aspects such as underlying mechanisms, therapeutic benefits, practical applications, and supporting research.
                - Use patent numbers as citations when discussing specific inventions like (US 20180307744). This citation is provided along with the contexts, and don't provide citations outside the provided contexts.\n
                - Do not hallucinate.\n
                - Do not include any irrelevant information. \n

                PATENT SUMMARIES: '''{state.patent_research_results}''' \n
                at the end of your report, provide a list of citations that only used in the report.
                """

    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1,
    )

    human_message_content = f"Create a Summary using the Context on this topic: \n <User Input> \n {topic} \n <User Input>\n\n"
    response = llm_model.invoke(
        [SystemMessage(content=patent_review_prompt),
         HumanMessage(content=human_message_content)]
    )

    state.patent_running_summary = f"## Summary\n{response.content}\n ## Sources:\n{state.patent_sources_gathered}"

    return {"patent_running_summary": state.patent_running_summary}


def article_deep_review(state: DeepSearchState):
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    patent_review_prompt = f"""
        You are an expert research assistant tasked with writing a comprehensive technical research report based on a provided collection of academic research articles. 
        Each article includes a summary of the title and and the abstract. 
        The report should be detailed, formal, and structured according to the following guidelines:
        
        🔹 Report Structure:
        
        - Abstract (Summary): 
            Provide a concise summary of the key themes, technologies and contributions across the research articles.
        
        - Background and Motivation:
            Explain the context, existing challenges, or prior research that the current studies aim to address.
        
        
        - Description: '''{topic}'''
            - Provide a detailed description of the articles that are most relevant to the topic.
            - Highlight the methods, approaches, devices, technologies, research questions, and objectives.
            - Identify the methodologies used and the problems these studies aim to solve.
            - Describe the findings or innovations introduced.
            - Trending sub topics.
        
        - Applications and Implications:
            Discuss potential real-world applications, theoretical impacts, or future research directions suggested by the articles.
       
        - future Direction:
            Provide a future direction of the research topic based on the provided context..
        
        - Conclusion:
            Summarize the overall contributions and insights gathered from the reviewed articles.
        
        
        🔹 Formatting and Style:
            Use formal, academic language appropriate for a research synthesis or literature review.
            Reference each article by its URL.
            Group similar or related studies where appropriate to avoid redundancy.
            Where useful, include tables, bullet points, or diagrams (optional, based on capabilities).
        
        - Requirements:
            - Use article URLs as citations when discussing specific studies. These URLs are provided along with the abstracts. 
            - Do not fabricate information or findings not present in the providing contexts.
            - Do not include any irrelevant or speculative information.
        
        RESEARCH ARTICLE ABSTRACTS: '''{state.article_research_results}'''
        """

    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(patent_review_prompt,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.6,
                                          top_k=5,
                                          temperature=0.2)
                                      )
    state.article_running_summary = f"## Summary\n{response.text}\n\n ## Sources:\n{state.article_sources_gathered}"

    return {"article_running_summary": state.article_running_summary}

