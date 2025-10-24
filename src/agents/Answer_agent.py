from dotenv import load_dotenv
import os

from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import openai

from src.agents.state import DeepSearchState

load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


def finalize_answer(state: DeepSearchState):
    """
    LangGraph node that compiles the final research answer.
        Produces the final answer then merging with proper citations.
    Args:
        state:

    Returns:

    """
    if 'gpt' in state.llm:
        return finalize_answer_by_openai(state)
    elif 'gemini' in state.llm:
        return finalize_answer_by_gemini(state)


def finalize_answer_by_gemini(state: DeepSearchState):
    answer_prompt = f"""You will be provided with patent passages as context to answer the question at the end. Please follow the following rules:
        Output Format:
             - Format your response as a JSON object with these exact keys:
             - "answer": the answer of the question
             - "sources": list of sources               
            
              Example 1:
                '''{{
                    "answer": "CAP kills cancer cells by triggering the rise of intracellular reactive oxygen species (ROS), DNA damage, mitochondrial damage, or cellular membrane damage (US20190231411A1). The rise of intracellular ROS always occurs in cancer cells upon CAP treatment, which causes a noticeable damage on the antioxidant system and subsequently DNA double strands break (DSB) to a fatal degree (US10479979B2). Serious DNA damage and other effect of CAP on cancer cells result in the cell cycle arrest, apoptosis or necrosis with a dose-dependent pattern (US10479979B2).. ...."
                    "sources": ["US20190231411A1", "US10479979B2"]
                }}'''
              Example 2:
                '''{{
                    "answer": "some medical and cosmetic applications are used for treating non-malignant skin growths, nail fungus infections, skin rejuvenation, and wrinkle removal. ...."
                    "sources": ["WO2022178164A1", "US20220256682A1", "US10479979B2", "US20220256682A1"]
                }}'''  
                 
        Requirements:
            - keep the answer concise.
            - ALWAYS return answer with a "sources" part in your answer.
            - For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences, like (US20220168565A1).
            - The "sources" is provided with each passage in the context like "US10023858B2", and do not change it. 
            - Use only the context to answer the question. 
            - If the answer is not in the provided context or passages, just say 'No Answer Found!
        
        Context:\n {state.patent_research_results}?\n
        Question: \n{state.research_topic}\n
        Answer:
    """
    llm_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.6,
        max_retries=2,
        api_key=gemini_api_key)

    result = llm_model.invoke(answer_prompt)

    return {
        "research_topic": state.research_topic,
        "answer": result.content,
        "patent_sources_gathered": state.patent_sources_gathered
    }


def finalize_answer_by_openai(state: DeepSearchState):

    answer_prompt = ("""You will be provided with patent passages as context to answer the question at the end. Please follow the following rules:
        Output Format:
             - Format your response as a JSON object with these exact keys:
             - "answer": the answer of the question
             - "sources": list of sources               

              Example 1:
                '''{{
                    "answer": "CAP kills cancer cells by triggering the rise of intracellular reactive oxygen species (ROS), DNA damage, mitochondrial damage, or cellular membrane damage (US20190231411A1). The rise of intracellular ROS always occurs in cancer cells upon CAP treatment, which causes a noticeable damage on the antioxidant system and subsequently DNA double strands break (DSB) to a fatal degree (US10479979B2). Serious DNA damage and other effect of CAP on cancer cells result in the cell cycle arrest, apoptosis or necrosis with a dose-dependent pattern (US10479979B2).. ...."
                    "sources": ["US20190231411A1", "US10479979B2"]
                }}'''
              Example 2:
                '''{{
                    "answer": "some medical and cosmetic applications are used for treating non-malignant skin growths, nail fungus infections, skin rejuvenation, and wrinkle removal. ...."
                    "sources": ["WO2022178164A1", "US20220256682A1", "US10479979B2", "US20220256682A1"]
                }}'''  

        Requirements:
            - keep the answer concise.
            - ALWAYS return answer with a "sources" part in your answer.
            - For each part of your answer, indicate which sources most support it via valid citation markers at the end of sentences, like (US20220168565A1).
            - The "sources" is provided with each passage in the context like "US10023858B2", and do not change it. 
            - Generate a high-quality answer to the user's question based on the provided context and the user's question. 
            - If the answer is not in the provided context or passages, just say 'No Answer Found!

        Context:\n 
        Question: 
        Answer:
    """
                     )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", answer_prompt),
        ("user", state.research_topic),
    ])
    llm = ChatOpenAI(model_name=openai_model,
                     temperature=0.6,
                     )
    pipeline = prompt_template | llm
    response = pipeline.invoke({"query": state.research_topic, "context": state.patent_sources_gathered})

    return {
        "research_topic": state.research_topic,
        "answer": response,
        "patent_sources_gathered": state.patent_sources_gathered
    }


if __name__ == "__main__":
    summary = f""" 
    # retrieved_patents# :# patent number# : # EP3869995A4# , # title# : # APPARATUS AND METHODS FOR PLASMA NAIL SURFACE TREATMENT# , # summary# : "The invention relates to the technical field of nail surface treatment, specifically improving the adhesion of energy-cured nail coatings.\n\nThe objective is to provide an apparatus and method that improves the adhesion of energy-cured nail coatings to the nail surface without damaging the nail. The technical problem is that current methods for improving adhesion, such as etching or chemical promoters, can weaken the nail or cause allergic reactions. The technical solution involves pretreating the nail surface with non-thermal plasma (NTP) before applying the coating.\n\nThe invention can be used in cosmetic industry for applying and curing energy cured nail coatings.\n\nThe core of the invention is an apparatus comprising: a non-thermal plasma generator, a curing lamp (LED, UV, or combined), electronic components for power and control, and a housing to receive at least the nail portion of a user# s digits.\n", # patent number# : # US20170216615A1# , # title# : # PLASMA TREATMENT OF AN INFECTED NAIL OR INFECTED SKIN# , # summary# : "- The invention is in the technical field of plasma treatment, specifically non-thermal plasma for treating infected nails or skin.\n- The objective is to provide an improved approach to plasma treatment for infections, overcoming drawbacks of prior art. The technical problem is that plasma treatment can dry out the nail or skin, reducing its efficacy. The technical solution involves applying plasma, rehydrating the nail or skin, and repeating the process while controlling the hydration level.\n- The invention can be used in medical, professional, or home environments to treat nail or skin infections, such as onychomycosis, athlete# s foot, and warts, and for lightening the color of nails.\n- The core of the invention is the use of non-thermal plasma for treating infected nails or skin, involving sequential application of plasma and rehydration, where the plasma is applied until the hydration level drops by at most 30 wt % based on the initial moisture content.\n", # patent number# : # EP4463089A1# , # title# : # DEVICES AND METHODS FOR TREATING SKIN TISSUE USING COLD PLASMA# , # summary# : # The patent document relates to the technical field of treating skin and nails using cold plasma. The invention aims to solve the problem of existing skin treatment methods that cause wounds and scarring by providing a non-invasive system and method for treating dermatological diseases and disorders, including skin and nail conditions, using cold plasma at skin temperature. The invention can be used for treating non-malignant skin growths and abnormalities, nail fungus infections, and for cosmetic treatments like skin rejuvenation and wrinkle removal. The core of the invention is a system for treating skin and nails with cold plasma, comprising a discharge device with a handle and applicator, and control infrastructure with a waveform generator, where the applicator has an elongated tube and a cathode, and the handle has a flyback amplifier, and the waveform generator induces the flyback amplifier to apply a voltage at the cathode to generate a self-sustaining Townsend avalanche when the tube is positioned near a target site, producing a cold plasma discharge without heating the target, and where the applicator is configured to couple to a vacuum system configured to reduce the pressure within the elongated tube such that, during application of plasma to the target site, the plasma is applied directly to the target site.\n# , # patent number# : # EP4173446A4# , # title# : # DEVICES AND METHODS FOR TREATING SKIN TISSUE USING COLD PLASMA# , # summary# : # The invention relates to the technical field of treating skin conditions using cold plasma. The objective is to provide devices and methods for treating dermatological diseases and disorders, including skin and nail conditions, using cold plasma at skin temperature, addressing issues like skin growths, abnormalities, and nail fungus infections without causing wounds or scarring. The invention can be used for dermatological, medical, and aesthetic treatments, such as skin rejuvenation and removal or softening of wrinkles, and is effective for both light and dark skin tones. The core of the invention is a system for treating skin and nails with cold plasma, comprising a discharge device with a handle and applicator, and control infrastructure with a waveform generator, where the applicator has an elongated tube and a cathode, the handle has a flyback amplifier, and the waveform generator induces the flyback amplifier to apply a voltage at the cathode to generate a self-sustaining Townsend avalanche, producing a cold plasma discharge with specific current and power levels to treat the target site without heating it.\n# , # patent number# : # US20230270968A1# , # title# : # SYSTEM AND PLASMA FOR TREATING AND/OR PREVENTING A VIRAL, BACTERIAL AND/OR FUNGAL INFECTION# , # summary# : "The invention relates to a system for treating and/or preventing viral, bacterial, or fungal infections in the oral cavity or respiratory tract using reactive species generated by plasma. The objective is to provide an improved treatment option for infections, especially viral epidemics. This is achieved by using a plasma source located outside the patient# s body and a species directing member with at least one duct to guide reactive species into the oral cavity and/or respiratory tract. The system can be used to treat infections in the nose, throat, trachea, or lungs. The core of the invention is a system for treating or preventing a viral, bacterial or fungal infection in the oral cavity or along the respiratory tract of a patient by reactive species generated by plasma, the system comprising: a plasma source generating reactive species in a gas, the plasma source being configured to be located outside a body of the patient; and a species directing member forming at least one duct to guide at least a part of the reactive species generated by the plasma source into at least one of the oral cavity and the respiratory tract, wherein the plasma source comprises a first electrode in the form of a laminar shaped high-voltage electrode.\n", # patent number# : # US9656095B2# , # title# : # Harmonic cold plasma devices and associated methods# , # summary# : # Here is a concise summary of the patent document you provided:\n\n-   **Technical Field:** The invention relates to cold plasma devices and methods, particularly hand-held devices for generating cold plasmas.\n\n-   **Objectives:** The invention addresses the need for improved cold plasma devices that overcome issues like electrode degradation and overheating, and that can deliver cold plasma plumes of different sizes and shapes for various applications. The technical solution involves a nozzle design that maintains a stable cold plasma plume while allowing for different aperture shapes and sizes, including the use of a foam material to support larger apertures.\n\n-   **Uses/Applications:** The invention is applicable to medical treatments, including wound healing, anti-bacterial processes, other medical therapies, sterilization, dermatology applications, skin cancer, dental caries, surgical site applications, diabetic ulcers, and internal treatments within bodily lumens or cavities using elongated devices.\n\n-   **Core of the Invention/Novelty:** The core of the invention is a cold plasma application device comprising a cold plasma generation device having a cold plasma outlet port; and a nozzle having a proximal aperture, a distal aperture and a solid wall located at the distal aperture, the proximal aperture configured to be coupled to the cold plasma outlet port to receive cold plasma from the cold plasma generation device, the distal aperture being non-perpendicular to a longitudinal axis of the nozzle, wherein cold plasma is directed by the distal aperture to a treatment area, and wherein the solid wall precludes cold plasma from contacting a non-treatment area.\n# , # patent number# : # US20130199540A1# , # title# : # Device for Plasma Treatment of Living Tissue# , # summary# : # - **Technical Field:** The invention relates to plasma medicine, specifically devices for plasma treatment of living tissue.\n\n- **Objectives:** The invention addresses the technical problem of achieving more reliable and faster plasma treatment of living tissue while preventing excessive thermal stress. The technical solution is a device that adjusts plasma output based on the position relative to the tissue.\n\n- **Uses/Applications:** The device can be used for disinfection of body parts, treatment of large-area skin complaints, cleaning of persons in contaminated protective suits, and pre/post-operative disinfection.\n\n- **Core of the Invention/Novelty:** The core of the invention is a device for plasma treatment of living tissue, comprising a plasma source, a support device for a body part, a movement device for the plasma source, and a control device, wherein the control device adjusts the plasma output as a function of the position relative to the tissue.\n# , # patent number# : # US8725248B2# , # title# : # Methods for non-thermal applications of gas plasma to living tissue# , # summary# : # The invention relates to the technical field of non-thermal plasma applications for living tissue treatment. The objective of the invention is to provide a method for treating living tissue with plasma without causing thermal damage, by employing tissue as an electrode of a high-voltage electrical discharge with relatively low total current and current density. This method can be used for blood coagulation, sterilization, disinfection, re-connection of tissue, and treatment of tissue disorders. The core of the invention is a method of generating a high-voltage non-thermal plasma discharge for treatment of living tissue comprising the step of passing sufficient electrical current to sustain the plasma discharge between the tissue and a surface of high-voltage electrode positioned proximate to the tissue and connected by electrical conductor to a power supply, and wherein a current density in said plasma is limited by the presence of a barrier insulator or a semiconductor positioned between the tissue and the electrode.\n# 
    """
    q = "How can nonthermal plasma, cold plasma or non-equilibrium plasma be applied for nail surface treatment?"
    # response = finalize_answer_by_openai(q, summary)
    # print(response)
