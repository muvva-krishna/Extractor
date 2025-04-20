from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
import io
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from rich import print
from ast import literal_eval
import os
from asset import system_prompt
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import ServerlessSpec
import uuid

load_dotenv()
api_key = ""
client = OpenAI(api_key=api_key)


def convert_doc_to_images(path):
    images = convert_from_path(path)
    return images


def get_img_uri(img):
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)

    base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')

    data_uri = f"data:image/png;base64,{base64_png}"
    return data_uri


def analyze_image(data_uri):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        }
                    ]
                },
            ],
            max_tokens=2000,
            temperature=0.1,
            top_p=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"



def analyze_doc_image(img):
    img_uri = get_img_uri(img)
    data = analyze_image(img_uri)
    return data

def process_all_pdfs_in_folder(folder_path):
    all_chunks = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_entry = {"pdf_name": pdf_file, "chunks": []}
        path = os.path.join(folder_path, pdf_file)
        print(f"Processing PDF: {pdf_file}")

        images = convert_doc_to_images(path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(analyze_doc_image, img) for img in images]
            ordered_results = []
            with tqdm(total=len(images), desc="Analyzing pages") as pbar:
                for future in futures:
                    result = future.result()
                    ordered_results.append(result)
                    pbar.update(1)
            pdf_entry["chunks"].extend(ordered_results)
            all_chunks.append(pdf_entry)

    return all_chunks


pinecone_api_key = ""
pc = Pinecone(api_key=pinecone_api_key)
pcindex = pc.Index(name = "extractor",host="https://extractor-vd1mwjl.svc.aped-4627-b74a.pinecone.io")

embeddings = OpenAIEmbeddings(api_key=api_key,model = "text-embedding-3-large")
vectorstore = PineconeVectorStore(index= pcindex,embedding= embeddings)

text_splitter = RecursiveCharacterTextSplitter()

def create_embeddings(documents):
    return embeddings.embed_documents([doc.page_content for doc in documents])

def store_embeddings(documents, embeddings, batch_size=100):
    pinecone_data = [
        {
            "id": f"{uuid.uuid4()}",
            "values": embedding,
            "metadata": {
                "text": doc.page_content,
                "pdf_name": doc.metadata["pdf_name"],
                "chunk_number": doc.metadata["chunk_number"]
            }
        }
        for doc, embedding in zip(documents, embeddings)
    ]

    for i in range(0, len(pinecone_data), batch_size):
        batch = pinecone_data[i:i + batch_size]
        res = pcindex.upsert(vectors=batch)
        print(f"[green]Upserted batch of {len(batch)} vectors. Pinecone response: {res}[/green]")


# Example usage in the main execution
if __name__ == "__main__":
    folder_path = "./dataset"
    labeled_chunks = process_all_pdfs_in_folder(folder_path)

    chunk_documents = []
    for pdf_entry in labeled_chunks:  # each PDF
        pdf_name = pdf_entry["pdf_name"]
        for i, chunk in enumerate(pdf_entry["chunks"]):
            if not chunk or "error" in chunk.lower():  # Filter empty/error chunks
                continue
            doc = Document(
                page_content=chunk,
                metadata={"pdf_name": pdf_name, "chunk_number": i}
            )
            chunk_documents.append(doc)

    print(f"[blue]Total valid chunks to embed: {len(chunk_documents)}[/blue]")


    # Create and store embeddings
    chunk_embeddings = create_embeddings(chunk_documents)
    store_embeddings(chunk_documents, chunk_embeddings, batch_size=20)

    print("[green]All documents stored in Pinecone with PDF metadata.[/green]")

