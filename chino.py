from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests
import os

# Load environment variables
load_dotenv()

# Function to convert PDF pages to images
def convert_pdf_to_images(file_path, scale=300/72):
    pdf_file = pdfium.PdfDocument(file_path)
    page_indices = list(range(len(pdf_file)))
    renderer = pdf_file.render(pdfium.PdfBitmap.to_pil, page_indices=page_indices, scale=scale)
    
    final_images = [{i: img} for i, img in zip(page_indices, renderer)]
    return final_images

# Function to extract text from images
def extract_text_from_img(images_list):
    image_content = [image_to_string(Image.open(BytesIO(list(img.values())[0].tobytes()))) for img in images_list]
    return "\n".join(image_content)

# Function to process URL-based PDF extraction
def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    return extract_text_from_img(images_list)

# Function to extract structured data using LLM
def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-4")  # Updated to GPT-4 for better accuracy
    template = """
    You are an expert in document extraction. Please extract relevant information based on these data points:
    {data_points}
    
    Content:
    {content}
    
    Return the extracted details as a JSON array:
    """
    prompt = PromptTemplate(input_variables=["content", "data_points"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(content=content, data_points=data_points)

# Function to send extracted data to Make.com webhook
def send_to_make(data):
    webhook_url = os.getenv("MAKE_WEBHOOK_URL", "https://hook.eu1.make.com/xxxxxxxxxxxxxxxxx")
    try:
        response = requests.post(webhook_url, json={"data": data})
        response.raise_for_status()
        print("Data sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data: {e}")

# Streamlit App
def main():
    st.set_page_config(page_title="Doc Extraction", page_icon=":bird:")
    st.header("Document Extraction :bird:")
    
    default_data_points = json.dumps({
        "invoice_item": "what is the item that was charged",
        "Amount": "how much does the invoice item cost in total",
        "Company_name": "company that issued the invoice",
        "invoice_date": "when was the invoice issued"
    }, indent=4)
    
    data_points = st.text_area("Data points", value=default_data_points, height=170)
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if uploaded_files and data_points:
        results = []
        for file in uploaded_files:
            with NamedTemporaryFile(delete=True, suffix='.pdf') as temp_file:
                temp_file.write(file.getbuffer())
                content = extract_content_from_url(temp_file.name)
                extracted_data = extract_structured_data(content, data_points)
                try:
                    json_data = json.loads(extracted_data)
                    results.extend(json_data if isinstance(json_data, list) else [json_data])
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing JSON: {e}")
        
        if results:
            df = pd.DataFrame(results)
            st.subheader("Results")
            st.data_editor(df)
            if st.button("Sync to Make"):
                send_to_make(results)
                st.success("Synced to Make!")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
