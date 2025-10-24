# Copyright Mustafa Sofean 2025 - FIZ-Karlsruhe

import os
import re

import requests
from requests.auth import HTTPBasicAuth
from vespa.application import Vespa
import pandas as pd

import xml.etree.ElementTree as ET
from src.utils.retrieval_utils import extract_values_from_json, get_json_array_value
from dotenv import load_dotenv

load_dotenv()

VESPA_ENDPOINT = os.getenv('VESPA_ENDPOINT')
P4S_SEARCH_API_ENDPOINT = os.getenv('P4S_SEARCH_API_ENDPOINT')
P4S_SEARCH_API_USER = os.getenv('P4S_SEARCH_API_USER')
P4S_SEARCH_API_PASSWORD = os.getenv('P4S_SEARCH_API_PASSWORD')
P4S_SEARCH_API_TOKEN_VALUE = os.getenv('P4S_SEARCH_API_TOKEN_VALUE')



def search_patent_passage(query: str, schema_name: str, rank_function: str = "lexical", hits: int = 20):
    """
    search in Vespa engine
    :param query:
    :param rank_function:
    :param hits:
    :return:
    """
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

    query = re.sub('[^a-zA-Z]', ' ', query)
    yql = None
    vespa_app = Vespa(url=VESPA_ENDPOINT)
    if rank_function == "lexical":
        yql = {
            "yql": "select ID, PNK, PASSAGE, SECTION from " + schema_name + " where userQuery() ",
            "query": query,
            "hits": hits,
            "ranking": {
                "profile": "bm25"
            },
            "type": "weakAnd"
        }

    results = vespa_app.query(body=yql)
    data = results.json['root']['children']
    id = extract_values_from_json(data, 'ID')
    pnk = extract_values_from_json(data, 'PNK')
    passage = extract_values_from_json(data, 'PASSAGE')

    df = pd.DataFrame(list(zip(id, pnk, passage)),
                      columns=['Passage ID', 'Patent No', 'PASSAGE'])
    #df['Patent No'] = df['Patent No'].apply(lambda x: get_pnk_by_id(x))

    df_to_json = df.to_json(orient='records')
    data = '{"data":' + df_to_json + '}'

    return data


def search_patent_doc(query: str, schema_name: str, rank_function: str = "lexical", hits: int = 20):
    """
    search in Vespa engine
    :param query:
    :param rank_function:
    :param hits:
    :return:
    """
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''

    query = re.sub('[^a-zA-Z]', ' ', query)
    yql = None
    vespa_app = Vespa(url=VESPA_ENDPOINT) #
    if rank_function == "lexical":
        yql = {
            "yql": "select ID, PNK,TIEN, ABEN, DETDEN, CLMEN, PD from " + schema_name + " where userQuery() and RFM=1",
            "query": query,
            "hits": hits,
            "ranking": {
                "profile": "bm25"
            },
            "type": "weakAnd"
        }
    results = vespa_app.query(body=yql)
    data = results.json['root']['children']
    # ids = extract_values_from_json(data, 'ID')
    pns = extract_values_from_json(data, 'PNK')
    tien = extract_values_from_json(data, 'TIEN')
    aben = extract_values_from_json(data, 'ABEN')
    patd = extract_values_from_json(data, 'PD')
    description = get_json_array_value(data, 'DETDEN')
    claims = get_json_array_value(data, 'CLMEN')
    df = pd.DataFrame(list(zip(pns, tien, aben, description, claims, patd)),
                      columns=['Patent No', 'Title', 'Abstract', 'Description', 'Claims', 'Publication Date'])

    df_to_json = df.to_json(orient='records')
    data = '{"data":' + df_to_json + '}'

    return data


def get_pnk_by_id(patentID: str):
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    search_url = VESPA_ENDPOINT+"?query=ID:" + patentID
    response = requests.get(search_url).json()
    data = response['root']['children']
    pnk = ""

    for record in data:
        pnk = record["fields"]["PNK"]
        return pnk


    return

def get_patent_info_from_vespa_index(field_name: str,
                                     field_value: str):
    """

    :param field_name:
    :param field_value:
    :return:
    """
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    search_url = VESPA_ENDPOINT + "search/?query=" + field_name + ":" + field_value
    response = requests.get(search_url).json()
    data = response['root']['children']
    ti = []
    ab = []
    detd = []
    clms = []
    pn = []
    pk = []

    for record in data:
        pn.append(record["fields"]["PN"])
        pk.append(record["fields"]["PK"])
        ti.append(record["fields"]["TIEN"])
        ab.append(record["fields"]["ABEN"])
        detd.append(record["fields"]["DETDEN"])
        clms.append(record["fields"]["CLMEN"])

    return pn, pk, ti, ab, detd, clms


def get_patent_data_by_stn_api(an: str, auth_token: str, patent_db: str):
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    url = P4S_SEARCH_API_ENDPOINT + "database/" + patent_db + "/document/raw/" + an
    auth = HTTPBasicAuth(P4S_SEARCH_API_USER, P4S_SEARCH_API_PASSWORD)
    headers = {
        "accept": "application/xml",
        'Authorization': 'Basic',
        'Token': auth_token
    }
    try:
        response = requests.get(url, auth=auth, headers=headers)

        return response.content.decode("utf-8")

    except requests.exceptions.RequestException as e:
        print(f"Error during calling STN-SAPI call: {e}")

    return None


def get_stn_sapi_token(token: str):
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    url = P4S_SEARCH_API_ENDPOINT + "database/INPADOC/status"
    auth = HTTPBasicAuth(P4S_SEARCH_API_USER, P4S_SEARCH_API_PASSWORD)
    headers = {
        "accept": "application/json",
        'Authorization': 'Basic',
        'Token': token
    }
    try:
        response = requests.get(url, auth=auth, headers=headers)
        token = response.headers.get("Token")
        if token:
            return token
        else:
            print("Failed to retrieve token. Check your credentials or API response format.")

    except requests.exceptions.RequestException as e:
        print(f"Error during STN-SAPI call: {e}")

    return None


def extract_patent_data_by_SAPI(an: str, patent_db):
    auth_token = get_stn_sapi_token(P4S_SEARCH_API_TOKEN_VALUE)
    sapi_response = get_patent_data_by_stn_api(an, auth_token, patent_db)
    """
    cms_doc = parseString(sapi_response)
    response = cms_doc.firstChild
    document_response = response.firstChild
    document = document_response.firstChild
    doc_member = document.firstChild
    #ti = get_element_content_by_attribute()
    print(doc_member.nodeValue)
    """

    response = ET.fromstring(sapi_response)
    document_response = response[0]
    document = document_response[0]
    document_members = document
    # print(len(document_members.getchildren()))

    pn, pk, title, abstract, description, claims = get_patent_member_info(document_members[0])

    return pn, pk, title, abstract, description, claims


def get_patent_member_info(document_member):
    pn = extract_xml_element_value(document_member, "PN")[0]
    title = extract_xml_element_value(document_member, "TIEN")[0]
    abstract = extract_xml_element_value(document_member, "ABEN")[0]
    description = extract_xml_element_value(document_member, "DETDEN")
    MCLMEN = extract_xml_element_value(document_member, "MCLMEN")
    CLMENINT = extract_xml_element_value(document_member, "CLMENINT")
    claims = MCLMEN + CLMENINT
    pk = extract_xml_element_value(document_member, "PK")[0]

    return pn, pk, title, abstract, description, claims


def extract_xml_element_value(document_node, field_name: str):
    value = [elem.text for elem in document_node.findall(".//*[@field=\'" + field_name + "\']")]
    return value

    # for doc in xml_tree.findall('.//document/*'):
    # print('doc')
    # value = doc.get('DETDEN')
    # print(value)


def get_element_content_by_attribute(root, tag, attr, value):
    for elem in root.findall(tag):
        if elem.get(attr) == value:
            return elem.text
    return None


if __name__ == "__main__":
    print("------")
    #print(search_patent_passage(query="cold plasma", schema_name="pt_passage"))
    print(search_patent_doc(query="cold plasma", schema_name="pt_doc"))
    # pn, pk, ti, ab, detd, clms = get_patent_info_from_vespa_index(field_name="ID", field_value="euRkfFMJdLjQ")
    # print(ab)
    # extract_patent_data_by_SAPI("2023196558") 2013196558
    # pn,pk, ti, ab, detd, clms = extract_patent_data_by_SAPI("2013196558", "USFULL")
    # print(pk)
    # print(search_patent_doc("cold plasma for tattoo removal", "plasma_doc", rank_function= "lexical", hits=5))
