from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
import io
import os
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
from dotenv import load_dotenv
from asset import system_prompt



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
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
            model="gpt-4.1",
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
            with tqdm(total=len(images), desc="Analyzing pages") as pbar:
                results = []
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

            # Maintain original image order
            ordered_results = [f.result() for f in futures]
            pdf_entry["chunks"].extend(ordered_results)

        all_chunks.append(pdf_entry)

    return all_chunks

if __name__ == "__main__":
    folder_path = "./dataset"
    result = process_all_pdfs_in_folder(folder_path)
    for pdf in result:
        print(f"\n[bold cyan]PDF: {pdf['pdf_name']}[/bold cyan]")
        for idx, chunk in enumerate(pdf['chunks']):
            print(f"\n[Page {idx + 1}]\n{chunk}\n")


