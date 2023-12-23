import argparse
import pickle
import requests
import xmltodict

from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()

def fetch_text_from_url(url):
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, features="html.parser")
    extracted_text = soup.get_text()

    cleaned_lines = (line.strip() for line in extracted_text.splitlines())
    return '\n'.join(line for line in cleaned_lines if line)


if __name__ == '__main__':
    sitemap_url = 'https://www.yourwebsite.com/sitemap.xml'

    sitemap_response = requests.get(sitemap_url)
    sitemap_xml = sitemap_response.text
    sitemap_data = xmltodict.parse(sitemap_xml)

    processed_pages = []
    for page_info in sitemap_data['urlset']['url']:
        page_url = page_info['loc']
        if page_url.startswith('https://'):
            processed_pages.append({'text': fetch_text_from_url(page_url), 'source': page_url})

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    document_chunks, metadata_list = [], []
    for processed_page in processed_pages:
        text_splits = text_splitter.split_text(processed_page['text'])
        document_chunks.extend(text_splits)
        metadata_list.extend([{"source": processed_page['source']}] * len(text_splits))
        print(f"Have splits {processed_page['source']} into {len(text_splits)} chunks")

    # Set OpenAI API key
    openAIEmbeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectore_store = FAISS.from_texts(document_chunks, openAIEmbeddings, metadatas=metadata_list)
    with open("vector_store.pkl", "wb") as file:
        pickle.dump(vectore_store, file)
