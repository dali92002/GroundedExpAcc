
from io import BytesIO
import os
import sys
import base64
import yaml
import requests
import cv2

from PIL import Image
from openai import OpenAI


API_CREDENTIALS_PATH = '/home/msouibgui/gpt.yaml'
MODEL_NAME = "gpt-4o-mini" 


# OpenAI API Endpoints
OPENAI_FILES_ENDPOINT = "https://api.openai.com/v1/files"
OPENAI_BATCHES_ENDPOINT = "https://api.openai.com/v1/batches"
OPENAI_RESULTS_ENDPOINT_TEMPLATE = "https://api.openai.com/v1/files/{file_id}/content"

# Debugging and Limiting Variables
LIMIT = None  # Set to None to process all samples, or set to an integer for testing

# Maximum size for each JSONL file in bytes (150MB)
MAX_JSONL_SIZE = 150 * 1024 * 1024

# Load OpenAI API Key
def load_api_key(credentials_path):
    with open(credentials_path, "r") as file:
        creds = yaml.safe_load(file)
    api_key = creds#.get('APIKEY')
    if not api_key:
        raise ValueError("APIKEY not found in the YAML file.")
    return api_key

# Encode Image to Base64
def encode_image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def creat_json_req(image, prompt, custom_id):
        
    image_base64 = encode_image_to_base64(image)

    # Create the JSON object
    request_object = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ChatGPT, a large language model trained by OpenAI."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },  
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
        }

    return request_object

def create_promt(question):
    return f"Do not use any previous memory, you are seing this image and this question for the first time. Do not provide an answer unless you find the supporting information on the document, random or knowledge based guesses are not allowed. Usually the answer exist in the document, take attention to find it. You are performing a DocVQA task. Answer the following question on the document image, provide only the answer (without any additional words, or introduction, or rewriting my question), an ANLS will be coputed to compare your response with a GT answer. Question: {question}"


# Upload JSONL File to OpenAI Files API
def upload_jsonl_file(api_key, file, json_id): 

    headers = { "Authorization": f"Bearer {api_key}" }
    files = { 'file': file }
    data = { 'purpose': 'batch' }
    print(f"Uploading {json_id} to OpenAI Files API...")
    response = requests.post(OPENAI_FILES_ENDPOINT, headers=headers, files=files, data=data) 
    if response.status_code != 200: 
        print(f"Failed to upload file {json_id}: {response.status_code} - {response.text}") 
        sys.exit(1) 
    file_info = response.json() 
    file_id = file_info['id'] 
    print(f"Uploaded {json_id}. File ID: {file_id}") 
    return file_id


def predict_gpt(img, question):
    api_key = load_api_key(API_CREDENTIALS_PATH)

    client = OpenAI(
        api_key = api_key,
        organization="org-r81J7whdTBtgrehjQGxSgBYC",
        project="proj_fk22qmeTAvws8jLifooTBiMC"
    )

    
    # create PIL image from cv2 image
    
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    image =  Image.fromarray(color_coverted) .convert('RGB')
    question = question
    prompt = create_promt(question)
    custom_id = "0"

    request_object = creat_json_req(image, prompt, custom_id)

    msg = request_object['body']['messages']
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        # messages=[{"role": "user", "content": "Say this is a test"}],
        messages=msg,
        stream=True,
    )

    answer = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            answer += chunk.choices[0].delta.content
    return  answer

