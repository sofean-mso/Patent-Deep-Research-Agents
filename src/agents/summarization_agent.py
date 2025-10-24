# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import os
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


from src.agents.state import DeepSearchState

load_dotenv()

import openai


import google.generativeai as genai

gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


def patent_summary_agent(ti: str,
                         ab: str,
                         detd: str,
                         clms: str,
                         model: str):
    if 'gpt' in model:
        return patent_summary_agent_by_openai(ti, ab, detd, clms)
    elif 'gemini' in model:
        return patent_summary_agent_by_gemini(ti, ab, detd, clms)


def patent_summary_agent_by_openai(ti: str,
                                   ab: str,
                                   detd: str,
                                   clms: str):
    """
    Summarize the patent document
    :param ti:
    :param ab:
    :param detd:
    :param clms:
    :return:
    """
    context = f"""
               Title:  {ti}
               Abstract: {ab}
               Description: {detd}
               Claims: {clms}
               """
    summary_prompt_template = f"""You will be provided with context includes Title, Abstract, Description, and Claims of a patent document.\n
        From only the provided texts, write a concise summary which includes: \n
            - the technical field or area of the invention, \n
            - the objectives of the invention,\n
            - the uses or applications of the invention, and,\n
            - the core of the invention or the novelty extracted from the first sentence of the claims.\n
            - the summary of the abstract, technical effects, technical problems, and technical means.\n
        
        Instructions:
            - Keep it concise.
            - Do not hallucinate.
            - Do not include any irrelevant information.
        context:  '''{context}'''\n
        CONCISE SUMMARY:
        """
    prompt_template = PromptTemplate(
        input_variables=["research_topic"],
        template=summary_prompt_template
    )
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.1,
    )
    summary_chain = prompt_template | llm_model
    result = summary_chain.invoke({"content": context})

    return result.content.replace('\n', '')


def patent_summary_agent_by_gemini(ti: str,
                                   ab: str,
                                   detd: str,
                                   clms: str):
    """
    Summarize patent by Gemini model
    :param ti:
    :param ab:
    :param detd:
    :param clms:
    :return:
    """

    context = f"""
               Title:  {ti}
               Abstract: {ab}
               Description: {detd}
               Claims: {clms}
               """
    summary_prompt_template = f"""You will be provided with Title, Abstract, Description, and Claims of a patent document.\n
        From only the provided patent texts, write a concise summary which includes: \n
            - the technical field or area of the invention, \n
            - the objectives of the invention,\n
            - the uses or applications of the invention, and,\n
            - the core of the invention or the novelty extracted from the first sentence of the claims.\n
            - the summary of the abstract, technical effects, technical problems, and technical means.\n

        
        Instructions:
            - Keep it concise.
            - Do not hallucinate.
            - Do not include any irrelevant information.
        
        Title:  '''{ti}'''\n
        Abstract: '''{ab}''' \n
        Description: '''{detd}'''\n
        Claims: '''{clms}''' \n
        CONCISE SUMMARY:
        """
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(summary_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.6,
                                          top_k=5,
                                          temperature=0.2)
                                      )
    time.sleep(5)

    return response.text


def article_summary_agent_by_gemini(ti: str,
                                    ab: str,
                                    body: str == ""
                                    ):
    """
    summarize paten with GPT model
    :param ti:
    :param ab:
    :param body:
    :return:
    """

    context = f"""
               Title:  {ti}
               Abstract: {ab}
               Body: {body}
               """
    summary_prompt_template = f"""
        Summarize the following research article's title and abstract in a concise and informative paragraph. 
        Focus on the main objective, methods, key findings, and significance.\n 
        Use clear and accessible language suitable for a general academic audience.\n
        Instructions:
            - Keep it concise.
            - Do not hallucinate.
            - Do not include any irrelevant information.

        Title:  '''{ti}'''\n
        Abstract: '''{ab}''' \n
        Body Text: '''{body}'''\n
        
        CONCISE SUMMARY:
        """
    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(summary_prompt_template,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.4,
                                          top_k=4,
                                          temperature=0.2)
                                      )
    time.sleep(5)

    return response.text


def summarize_patent_summary(state: DeepSearchState):
    """LangGraph node that summarizes patent research results.

    Uses an LLM to create or update a running summary based on the newest patent research
    results, integrating them with any existing summary.
    Args:
        state: Current graph state containing research topic, running summary,
              and patent research results
    Returns:
        Dictionary with state update, including running_summary key containing the updated summary
    """
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    # Existing summary
    existing_summary = state.patent_running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Context> \n {most_recent_web_research} \n <New Context>"
            f"Update the Existing Summary with the New Context on this topic: \n <User Input> \n {topic} \n <User Input>\n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_research} \n <Context>"
            f"Create accurate Summary using the Context on this topic: \n <User Input> \n {topic} \n <User Input>\n\n"
        )

    model = genai.GenerativeModel(gemini_model,
                                  )
    response = model.generate_content(human_message_content,
                                      generation_config=genai.types.GenerationConfig(
                                          candidate_count=1,
                                          top_p=0.5,
                                          top_k=5,
                                          temperature=0.1)
                                      )
    running_summary = response.text

    return {"running_summary": running_summary}


if __name__ == "__main__":
    # response = summarize_patent_summary(None)
    response = patent_summary_agent_by_openai(
        ti="PLASMA TORCH HEAD, PLASMA TORCH SHAFT AND PLASMA TORCH",
        ab="The invention relates to a plasma torch head, plasma torch shaft and a plasma torch that permit the plasma torch head to be simply and rapidly replaced.",
        detd="  [DESC0001]  'Electronic torch head, electronic torch shank and electronic torch    [DESC0002] The available invention concerns electronic torch head, comprehensively at least one fluid passage, an electrode, a nozzle, a powerline and a bearing surface on an edition side, electronic torch shank, comprehensively at least one Zuführungsleitung for gas, current supply line, at least one fluid passage, a powerline and a bearing surface on an edition side and electronic torch with at least one Zuführungsleitung for a gas, an electrode, a nozzle and a current supply line, whereby the electronic torch an electronic torch shank, which contains at least a first fluid passage, a first powerline and a first bearing surface on an edition side, and an electronic torch head covers, which contains at least a second fluid passage, a second powerline and a second bearing surface on an edition side, whereby first and second bearing surfaces rest upon axially to each other and those that at least first fluid passage with that at least second fluid passage in fluid connection are located as well as the first powerline with the second powerline in electrical connection stand,      [DESC0003] There is electronic torches well-known, which consist of an electronic torch shank and an electronic torch head, which are connectable with one another with a quick change catch. In the electronic torch head are the parts of the electronic torch, which must be worn in consequence of the enterprise fast and replaced more frequently. These are above all the electrode, the nozzle and the protective cap. In addition, in case of changing targeted applications of the plasma procedure, for example between cutting structural steel and cutting high-grade steel, the change of an electronic torch cutter head can be necessary on another. In order to realize this fast, a quick change catch is meaningful.      [DESC0004] the rear G 081 32 660 is described an electronic torch, which consists of an electronic torch shank, a plug-on connecting piece and an electronic torch head. The electronic torch has a locking pin protruding from the clutch surface and one on the opposite clutch surface according to trained drilling, into which the locking pin is importable during exact radial attitude. The electronic torch shank and the connecting piece are connected by means of a case adjustable on the electronic torch shank with guide pins, which are importable into appropriate axially and radially trained guide grooves in the connecting piece, bayonet-like under axial pressing with radial movement of the case. Both during manual handling and with automated systems it is pedantic to contrive first the locking pin into the drilling in order to then connect the other contacts for the supply and supply. Besides a damage cannot be excluded.      [DESC0005] an adjustment device and a procedure for an arc electronic torch system are descriptive the rear DE 695 11 728 T2. The arc electronic torch consists of an arc electronic torch shank and an arc electronic torch head. A total positioning guidance is used, in order to align the arc electronic torch at first for an admission. The admission can be a tapered edge. The admission has two passages with a Aufhahmeende and a top side, which are so dimensioned that orientation pins with a diameter are taken up. The orientation pins have at the same time openings, which can let a gas or a liquid through. The surface diameter is larger than the passage diameter and can adjust thereby slight misalignments. A middle passage is just as dimensioned and can also a gas or a liquid through-leads. During false positioning it can come to damages of the orientation pins, if after contriving the middle passage a Kraft in axial direction of the arc electronic torch works. It can lead when simultaneous use the orientation pins as passage for a gas or a liquid to leakages. Damages at the orientation pins make later positioning more difficult and brought together the components of the arc electronic torch, in particular if a small tolerance of the axles of the arc electronic torch head and the admission are demanded.      [DESC0006] In addition in principle the Ineinanderführen of two cylindrical body of an electronic torch is well-known. There is however the danger of an incorrect allocation of the connections and/or a damage the same.      [DESC0007] Additionally a very high centricity of the connection is often required. Then the play between Innenund a tip cylinder must be very small. This again makes then a joining more difficult.      [DESC0008] The invention is thus the basis the task to make a quick change possibility available for the electronic torch head.      [DESC0009] According to invention this task is solved with the electronic torch head of the kind initially specified by the fact that it exhibits a cylinder wall with an exterior surface and an annulus surface on its edition side, whereby on the exterior surface circulating more nver resembles radial recesses and nvor same radial projections/leads is intended, whereby more nver, nvor ≥ 0 and + nvor ≥ 5 is more nver, and in case of of n = 5 the sum of two neighbouring central angles, under which the projections/leads and/or recesses and/or a projection/lead and a recess transferred arranged with one another are, are not &gt; 180° and are of various sizes the five the central angles, or in case of of n &gt; 5 the sum of two neighbouring central angles, under those the projections/leads and/or recesses and/or a projection/lead and a recess to each other are transferred arranged, &gt; 180° are not and the n &gt; 5 central angles are of various sizes or at least two of the n &gt; 5 central angles are equal in size, whereby then the sum of the respective doubles occurring central angle and that in each case of it to both sides neighbouring central angles &lt; 180° is.      [DESC0010] The powerline can be in the fluid passages in an integrated way and/or separately implemented.      [DESC0011] Usually at least three fluid passages, i.e. intended for the supply of gas, are like orifice gas, and Zuund return pipe by cooling agent.      [DESC0012] Besides this task is solved with the electronic torch shank of the kind initially specified by the fact that it ufweist on its edition side a cylinder wall with an inner surface, whereby on the inner surface circulating nVor resembles radial projections/leads and more nver same radial recesses is intended, whereby nvor, ≥ 5 is more nver ≥ 0 and nVor + more nver, and in case of of n = 5 the sum of two neighbouring central angles, under which the projections/leads and/or recesses and/or a projection/lead and a recess transferred arranged with one another are, are not &gt; 180° and are of various sizes the five the central angles, or in case of of n &gt; 5 the sum of two neighbouring central angles, under those the projections/leads and/or recesses and/or a projection/lead and a recess to each other are transferred arranged, &gt; 180° are not and the n &gt; 5 central angles are of various sizes or at least two of the n &gt; 5 central angles are equal in size, whereby then the sum of the respective doubles occurring central angle and that in each case of it to both sides neighbouring central angles &lt; 180° is.      [DESC0013] With the electronic torch heads and - shanks can it itself around Plasmaschneidoder plasma welding heads and/or - shanks to act. Further this task is solved with the electronic torch of the kind initially specified by the fact that one of the electronic torch shank and the electronic torch head on its edition side a first cylinder wall with an exterior surface and an annulus surface exhibits as well as an outside diameter D21a and the other one from the electronic torch shank and the electronic torch head on its edition side exhibits a second cylinder wall with an inner surface and an inside diameter of D31a, whereby D31a is &gt; D21a and on the inner surface circulating nvor same radial projections/leads and more nver resembles radial recesses, whereby nvor, more nver ≥ 0 and nvor + more nver = n &gt; 5 is, and on the exterior surface a same number of corresponding recesses standing thereby in interference and/or projections/leads, furthermore those is intended Projections/leads and recesses are so arranged that when connecting the electronic torch shank with the electronic torch head only the projections/leads and recesses in interference be brought must, before the first bearing surface and the second bearing surface come to resting upon, and in case of of n = 5 the sum of two neighbouring central angles, under which the projections/leads and/or recesses and/or a projection/lead and a recess are to each other transferred arranged, are not &gt; 180° and are of various sizes the five the central angles, or in case of of n &gt; 5 the sum of two neighbouring central angles, under which the projections/leads and/or recesses and/or a projection/lead and a recess are to each other transferred arranged, is not &gt; 180° and the n &gt; 5 central angles of various sizes or at least two are the n &gt; 5 central angles are equal in size, whereby then the sum of the respective doubles occurring central angle and that in each case of it to both sides neighbouring central angles &lt; 180° is.      [DESC0014] The electronic torch can be a Plasmaschneidoder - torches.      [DESC0015] With the electronic torch head it can be intended that the sum of two neighbouring central angles is &lt; 170°. Thus a still more stable plant is reached by annulus surface and projections/leads in the adding position. In accordance with a special execution form of the invention n = 5 is not and repeats itself the sum of two neighbouring central angle.      [DESC0016] In accordance with a further special execution form the electronic torch head contains four fluid passages.      [DESC0017] Favourable way is at least a fluid passage provided with a plug that.      [DESC0018] The powerline is likewise provided appropriately with a plug.      [DESC0019] Further it can be intended that recesses are rectangular slots. Of course the slots can exhibit also any other shape, for example arc-shaped, triangular etc.      [DESC0020] In accordance with a further special execution form ≥ 5. is more nver.      [DESC0021] Alternatively it is also conceivable that nvor ≥ 5 is.      [DESC0022] The Unteransprüche to the electronic torch shank concern favourable further educations the same.      [DESC0023] For example can on the inner surface of the cylinder wall in the direction of the edition side before the projections/leads a circulating radially outward extending chamfers myself to be intended. Joining is facilitated by these, since at the beginning of joining a larger diameter is available. The Unteransprüche to the electronic torch concern favourable further educations of the same.      [DESC0024] For example a special execution form of the electronic torch is characterized by that the electronic torch head exhibits the first cylinder wall and the second cylinder wall exhibits electronic torch shank.      [DESC0025] The invention is the basis the surprising realization that by the special number and arrangements by projections/leads and corresponding recesses simple and fast joining of electronic torch head and electronic torch shank without tilting is obtained. The annulus surface must be brought simply only to the plant at the projections/leads, i.e. into an adding position be brought, and afterwards relative to the projections/leads be turned, until the adding position is reached, with which at axially working Kraft the projections/leads and recesses with one another into interference steps. This is particularly favourable in situations, in which the electronic torches are not optically accessible clamped and. The quick change of the electronic torch head can take place as it were blindly.      [DESC0026] Beyond that the invention offers a quick change connection between electronic torch head and electronic torch shank with twist protection, to small tolerance between the axles of the electronic torch head and electronic torch shank and high centricity.      [DESC0027] The fluid passages both for the gas, and Plasmaund secondary gas, and for the cooling agent, can be used also for the current transmission.      [DESC0028] Further characteristics and advantages of the invention result from the requirements and the following description, in which two remark examples are in detail described on the basis the schematic designs. Shows: Figure 1 a side view of a front part of a plasma cutting torch before the Zusammenfugen of plasma cutting torch head and plasma cutting torch shank in accordance with a special execution form of the available invention partly on average;      [DESC0029] Figure 2 a side view of the front part of the plasma cutting torch during joining the plasma cutting torch head and the plasma cutting torch shank in the adding position partly on average;  ,",
        clms="  [CLM0001] . Electronic torch head (2), comprehensively at least exhibits one fluid passage, an electrode, a nozzle, a powerline and a bearing surface on an edition side, thereby characterized,      [CLM0002]  more daßer on its edition side a cylinder wall (21) with an exterior surface (21a) and an annulus surface (22), whereby on the exterior surface (21a) circulating more nver resembles radial recesses and nvor same radial projections/leads are intended, whereby more nVer, nvor - ® <sup>UnC^</sup> + HVor ≥ 5 is <sup>more nVer</sup>, Undim      [CLM0003]  case of n = 5 the sum of two neighbouring central angles (α and ß or ß and γ or γ and δ or δ and ε or ε and α), under those the projections/leads and/or recesses and/or a projection/lead and a recess transferred are with one another arranged, &gt; 180° are not and the five the central angles (α, ß, γ, δ, ε) are of various sizes,      [CLM0004]  oderim      [CLM0005]  case of n &gt; 5 the sum of two neighbouring central angles (α and ß or ß and γ or γ and δ or δ and ε or ε and α), under which the projections/leads and/or recesses and/or a projection/lead and a recess are to each other transferred arranged, not &gt; 180° is and the n &gt; 5 central angles (α, ß, γ, δ, ε) are of various sizes or at least two of the n &gt; 5 central angles (α, ß, γ, δ, ε) are equal in size, whereby then in each case the sum of the respective doubles occurring central angle (α or ß or γ or δ or ε) and that of it to both sides neighbouring central angles (α and γ or ß and δ or γ and ε or δ and α or ε and ß) &lt; 180° is.      [CLM0006] 2. Electronic torch head (2) according to requirement 1, by characterized that the sum of two neighbouring central angles (α and ß or ß and γ or γ and δ or δ and ε or ε and α) &lt; 170° ist.3      [CLM0007] . Electronic torch head (2) according to requirement 1 or 2, by characterized that n = 5 is and itself the sum second neighbouring central angle (α and ß or ß and γ or γ and δ or δ and ε or ε and α) not wiederholt.4      [CLM0008] . Electronic torch head (2) after one of the preceding requirements, thereby characterized that it four fluid passages enthält.5      [CLM0009] . Electronic torch head (2) after one of the preceding requirements, thereby characterized that at least with a plug (241, 242, 243, 244) ist.6 provide a fluid passage      [CLM0010] . Electronic torch head (2) after one of the preceding requirements, thereby characterized that with a plug (245) ist.7 provide the powerline      [CLM0011] . Electronic torch head (2) after one of the preceding requirements, thereby characterized that the recesses rectangular slots (231, 232, 233, 234, 235) sind.8      [CLM0012] . Electronic torch head after one of the preceding requirements, thereby characterized that more nver ≥ 5 ist.9      [CLM0013] . Electronic torch head (2) after one of the requirements 1 to 7, by characterized that nvor ≥ 5 is.      [CLM0014] 10. Electronic torch shank (3), comprehensively at least one Zuführungsleitung for gas, current supply line, at least one fluid passage, a powerline and a bearing surface on an edition side,      [CLM0015]  thereby characterized, more daßer      [CLM0016]  on its edition side a cylinder wall (31) with an inner surface (31a) exhibits, whereby on the inner surface (31a) circulating nvor resembles radial projections/leads and more nver same radial recesses are intended, whereby nvor, ≥ 5 is more nver ≥ 0 and nVor + more iWer, undim      [CLM0017]  case of n = 5 the sum of two neighbouring central angles (α and ß or ß and γ or γ and δ or δ and ε or ε and α), under those those Projections/leads and/or recesses and/or a projection/lead and a recess transferred are with one another arranged, &gt; 180° are not and the five the central angles (α, ß, γ, δ, ε) are of various sizes,      [CLM0018]  oderim      [CLM0019]  case of n &gt; 5 the sum of two neighbouring central angles (α and ß or ß and γ or γ and δ or δ and ε or ε and α), under which the projections/leads and/or recesses and/or a projection/lead and a recess are to each other transferred arranged, not &gt; 180° is and the n &gt; 5 central angles (α, ß, γ, δ, ε) are of various sizes or at least two of the n &gt; 5 central angles (α, ß, γ, δ, ε) are equal in size, whereby then in each case the sum of the respective doubles occurring central angle (α or ß or γ or δ or ε) and that of it to both sides neighbouring central angles (α and γ or ß and δ or γ and ε or δ and α or ε and ß) &lt; 180° is.      [CLM0020] 11. Electronic torch shank (3) according to requirement 10, by characterized that the sum of two neighbouring central angles (α and ß or ß and γ or γ and δ or δ and ε or ε and α) &lt; 170° ist.12      [CLM0021] . Electronic torch shank (3) according to requirement 10 or 11, by characterized that n = 5 is and itself the sum second neighbouring central angle (α and γ or ß and δ or γ and ε or ε and ß) not wiederholt.13      [CLM0022] . Electronic torch shank (3) after one of the requirements 10 to 12, by characterized that additionally a supply line for secondary gas is intended and it four fluid passages enthält.14  "
    )
    # print(response)
