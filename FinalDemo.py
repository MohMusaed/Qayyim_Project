import os
import json
import requests
import pathlib
import textwrap
import PIL.Image
import pdfplumber
import re
import pandas as pd
from io import BytesIO
from google.cloud import vision
from google.oauth2 import service_account
from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import openai
import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TextSplitter
import base64
from chromadb.config import Settings
from langchain.embeddings.openai import OpenAIEmbeddings
import io
import datetime


# Set API keys
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Retrieve the JSON string from the environment variable
json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")

# Check if the environment variable is set
if not json_str:
    st.error("Google credentials not found in environment variables.")
    st.stop()  # Stop execution if credentials are not found
else:
    # Parse the JSON string into a dictionary
    credentials_info = json.loads(json_str)
    # Create credentials object
    credentials = service_account.Credentials.from_service_account_info(credentials_info)

# Streamlit app UI
st.set_page_config(page_title="تقرير الحادث المروري", page_icon="🚗")

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo&display=swap');
    html, body, [class*="css"]  {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        padding: 0.5em 2em;
        border-radius: 8px;
        border: none;
        font-size: 1em;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* New CSS for image uploaders and images */
    .image-uploader {
        background-color: #ffffff;
        padding: 1em;
        border: 2px dashed #ccc;
        border-radius: 10px;
        margin-bottom: 1em;
        text-align: center;
    }
    .image-uploader:hover {
        background-color: #f9f9f9;
        border-color: #aaa;
    }
    .image-container {
        background-color: #ffffff;
        padding: 1em;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        margin-bottom: 1em;
        text-align: center;
    }
    .uploaded-image {
        max-width: 100%;
        border-radius: 10px;
        border: 1px solid #e6e6e6;
    }
    </style>
    """, unsafe_allow_html=True)



st.title("🚗 تقرير الحادث المروري")
st.header("Qayyim")

st.markdown("###")

st.write("الرجاء تحميل الصوره وإدخال وصف الحادث.")

ol1, col2, col3 = st.columns(3)

# Arrange the file uploaders in columns with decorations
st.markdown("###")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="image-uploader">', unsafe_allow_html=True)
    accident_image_file = st.file_uploader("📸 صورة الحادث", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="image-uploader">', unsafe_allow_html=True)
    vehicle_reg_image1_file = st.file_uploader("🚗 استمارة السيارة الأولى", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="image-uploader">', unsafe_allow_html=True)
    vehicle_reg_image2_file = st.file_uploader("🚙 استمارة السيارة الثانية", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

# Input descriptions for each party
st.header("📝 وصف الحادث من الأطراف")

FirstPartyDescription = st.text_area("وصف الطرف الأول:")
SecondPartyDescription = st.text_area("وصف الطرف الثاني:")

current_datetime = datetime.datetime.now()

if accident_image_file is not None:
    # Display the uploaded image
    st.subheader("📸 صورة الحادث:")
    # st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(accident_image_file, caption='صورة الحادث', use_column_width=True, channels="RGB")
    # st.markdown('</div>', unsafe_allow_html=True)

    # Read the image data
    #image_data = accident_image_file.getvalue()

    # # Read the image and encode it in base64
    base64_image = base64.b64encode(accident_image_file.getvalue()).decode('utf-8')

    # Prepare the payload for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    prompt_arabic = """
    انت محقق حوادث مرورية أو شرطي مرور. سيتم تزويدك بصورة؛ إذا وجدت بها حادث، قم بتحديد الطرف الأول على يسار الصورة والطرف الثاني على يمين الصورة.
    أريد منك وصفًا للحادث وتحديد الأضرار إن وجدت فقط.
    وإن لم يكن هناك حادث في الصورة، قم بكتابة "لم يتم العثور على حادث في الصورة".
    """
    payload = {
      "model": "gpt-4o-mini",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt_arabic
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    # Send the request to the OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    # Check the response
    response_json = response.json()
    if 'choices' in response_json and len(response_json['choices']) > 0:
        message_content = response_json['choices'][0]['message']['content']
        st.write("**وصف الحادث:**")
        st.write(message_content)
    else:
        st.write("لم يتم العثور على وصف للحادث في الاستجابة.")


    AccidentDescription = message_content



# Function to detect text using Google Vision API
def detect_text(image_file):
    """Detects text in the file at the given image path."""
    # Initialize Google Cloud Vision Client with credentials
    client = vision.ImageAnnotatorClient(credentials=credentials)
    content = image_file.getvalue()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    full_text = texts[0].description if texts else ''
    return full_text

# (The rest of your label extraction functions remain the same)

# Main function to process the text and call each field extraction function
def format_vehicle_registration_text(detected_text):
    lines = [line.strip() for line in detected_text.split('\n') if line.strip()]

    label_indices, label_set = create_label_indices(lines)

    output_lines = [
        extract_owner_name(lines, label_indices),
        extract_owner_id(lines, label_indices),
        extract_chassis_number(lines, label_indices),
        extract_plate_number(lines, label_indices, label_set),
        extract_vehicle_brand(lines, label_indices),
        extract_vehicle_weight(lines, label_indices),
        extract_registration_type(lines, label_indices),
        extract_vehicle_color(lines, label_indices),
        extract_year_of_manufacture(lines, label_indices)
    ]

    return "\n".join(output_lines)

def create_label_indices(lines):
    label_indices = {}
    label_set = set()
    for idx, line in enumerate(lines):
        line_clean = line.strip(':').strip()
        # Map labels to their indices
        labels = [
            'المالك', 'هوية المالك', 'رقم الهوية', 'رقم الهيكل', 'رقم اللوحة',
            'ماركة المركبة', 'الماركة', 'الوزن', 'نوع التسجيل', 'طراز المركبة',
            'الموديل', 'حمولة المركبة', 'اللون', 'سنة الصنع', 'اللون الأساسي'
        ]
        for label in labels:
            if label in line_clean:
                label_indices[label] = idx
                label_set.add(label)
    return label_indices, label_set

def extract_owner_name(lines, label_indices):
    label = 'المالك'
    exclusion_list = ['القامة', 'سرباع', 'وزارة الداخلية', 'رخصة سير', 'التعامل مع الهوية']
    if label in label_indices:
        idx = label_indices[label]
        name = lines[idx].split(':')[-1].strip()
        if not name and idx + 1 < len(lines):
            name = lines[idx + 1].strip()
        if name:
            return f"المالك: {name}"
    else:
        # If label not found, try to find the owner's name in other lines
        for line in lines:
            line_clean = line.strip()
            if (re.match(r'^[\u0621-\u064A\s]{3,}$', line_clean) and
                line_clean not in exclusion_list and
                'وزارة' not in line_clean and
                'رخصة' not in line_clean and
                'التعامل' not in line_clean and
                len(line_clean.split()) > 2):
                return f"المالك: {line_clean}"
    return "المالك: غير متوفر"

def extract_owner_id(lines, label_indices):
    labels = ['هوية المالك', 'رقم الهوية', 'رقم السجل']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            id_line = lines[idx]
            owner_id = re.search(r'\b\d{10}\b', id_line)
            if not owner_id and idx + 1 < len(lines):
                id_line_next = lines[idx + 1]
                owner_id = re.search(r'\b\d{10}\b', id_line_next)
            if owner_id:
                return f"هوية المالك: {owner_id.group()}"
    # Fallback: search all lines for a 10-digit number
    for line in lines:
        owner_id = re.search(r'\b\d{10}\b', line)
        if owner_id:
            return f"هوية المالك: {owner_id.group()}"
    return "هوية المالك: غير متوفر"

def extract_chassis_number(lines, label_indices):
    labels = ['رقم الهيكل']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            chassis_line = lines[idx]
            match = re.search(r'رقم الهيكل[:\s]*(\S+)', chassis_line)
            if match:
                chassis_number = match.group(1)
                return f"رقم الهيكل: {chassis_number}"
            elif idx + 1 < len(lines):
                chassis_number = lines[idx + 1].strip()
                return f"رقم الهيكل: {chassis_number}"
        else:
            # Fallback: look for a line with 17-character alphanumeric string
            for line in lines:
                if re.match(r'^[A-HJ-NPR-Z0-9]{17}$', line):
                    return f"رقم الهيكل: {line.strip()}"
    return "رقم الهيكل: غير متوفر"

def extract_plate_number(lines, label_indices, label_set):
    labels = ['رقم اللوحة']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            plate_line = lines[idx]
            match = re.search(r'رقم اللوحة[:\s]*(.*?)(?:رقم|$)', plate_line)
            if match:
                plate_info = match.group(1).strip()
                plate_info = re.split(r'\s*رقم', plate_info)[0].strip()
                return f"رقم اللوحة: {plate_info}"
            else:
                plate_numbers = []
                next_idx = idx + 1
                while next_idx < len(lines):
                    next_line = lines[next_idx]
                    if next_line.strip(':').strip() in label_set:
                        break
                    if next_line:
                        plate_numbers.append(next_line)
                    next_idx += 1
                if plate_numbers:
                    return f"رقم اللوحة: " + ' '.join(plate_numbers)
    return "رقم اللوحة: غير متوفر"

def extract_vehicle_brand(lines, label_indices):
    labels = ['ماركة المركبة', 'الماركة']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            brand_line = lines[idx]
            match = re.search(r'(?:ماركة المركبة|الماركة)[:\s]*(.*)', brand_line)
            if match:
                brand = match.group(1).strip()
                if not brand and idx + 1 < len(lines):
                    brand = lines[idx + 1].strip()
                return f"ماركة المركبة: {brand}"
    known_brands = ['فورد', 'تويوتا', 'نيسان', 'هوندا', 'شيفروليه', 'مرسيدس']
    for line in lines:
        for brand in known_brands:
            if brand in line:
                return f"ماركة المركبة: {brand}"
    return "ماركة المركبة: غير متوفر"

def extract_vehicle_weight(lines, label_indices):
    labels = ['الوزن']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            weight_line = ' '.join(lines[idx:idx+2])
            weight_match = re.search(r'الوزن[:\s]*(\d+)', weight_line)
            if weight_match:
                return f"وزن المركبة: {weight_match.group(1)}"
    return "وزن المركبة: غير متوفر"

def extract_registration_type(lines, label_indices):
    label = 'نوع التسجيل'
    if label in label_indices:
        idx = label_indices[label]
        reg_line = lines[idx]
        match = re.search(r'نوع التسجيل[:\s]*(.*)', reg_line)
        if match:
            reg_type = match.group(1).strip()
            return f"نوع التسجيل: {reg_type}"
        elif idx + 1 < len(lines):
            reg_type = lines[idx + 1].strip()
            return f"نوع التسجيل: {reg_type}"
    return "نوع التسجيل: غير متوفر"

def extract_vehicle_color(lines, label_indices):
    labels = ['اللون', 'اللون الأساسي']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            color_line = lines[idx]
            match = re.search(r'(?:اللون|اللون الأساسي)[:\s]*(.*)', color_line)
            if match:
                color = match.group(1).strip()
                color = color.replace('الأساسي:', '').strip()
                return f"اللون: {color}"
            elif idx + 1 < len(lines):
                color = lines[idx + 1].strip()
                return f"اللون: {color}"
    return "اللون: غير متوفر"

def extract_year_of_manufacture(lines, label_indices):
    labels = ['سنة الصنع']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            year_line = ' '.join(lines[idx:idx+2])
            year_match = re.search(r'سنة الصنع[:\s]*(\d{4})', year_line)
            if year_match:
                return f"سنة الصنع: {year_match.group(1)}"
            else:
                year_match = re.search(r'\b(19|20)\d{2}\b', year_line)
                if year_match:
                    return f"سنة الصنع: {year_match.group(0)}"
    return "سنة الصنع: غير متوفر"

# Process vehicle registration images
if vehicle_reg_image1_file and vehicle_reg_image2_file:
    # Detect and format text for both images
    with st.spinner("جاري معالجة صور تسجيل المركبات..."):
        detected_text1 = detect_text(vehicle_reg_image1_file)
        formatted_text1 = format_vehicle_registration_text(detected_text1)

        detected_text2 = detect_text(vehicle_reg_image2_file)
        formatted_text2 = format_vehicle_registration_text(detected_text2)

    # Display the formatted text for both vehicle registrations
    st.write("### معلومات تسجيل السيارة الأولى:")
    st.text(formatted_text1)

    st.write("### معلومات تسجيل السيارة الثانية:")
    st.text(formatted_text2)
else:
    st.write("يرجى تحميل صور تسجيل المركبتين لاستخراج التفاصيل.")


@st.cache_resource
def load_data():
  return Chroma(persist_directory="./database", embedding_function=OpenAIEmbeddings())




# Function to generate the accident report
def generate_accident_report_with_fault(FirstPartyDescription, SecondPartyDescription, AccidentDescription, VehicleRegistration1, VehicleRegistration2):
    retrieved_context = load_data()

    result = retrieved_context.similarity_search(AccidentDescription + FirstPartyDescription + SecondPartyDescription)


    # Create the prompt dynamically by including the vehicle information and accident description
    prompt = (
        f"""
        وقت طباعة التقرير {current_datetime}
        
        يوجد حادث مروري لسيارتين:
        وصف الحادث بناءً على الصورة المقدمة: {AccidentDescription}

        تسجيل السيارة الأولى: {VehicleRegistration1}
        تسجيل السيارة الثانية: {VehicleRegistration2}

        وصف الطرف الأول: {FirstPartyDescription}
        وصف الطرف الثاني: {SecondPartyDescription}

        بناءً على القوانين المرورية والمعلومات التالية التي تم استرجاعها:
        {result}

        أريد منك أن تكتب تقريرًا كاملاً عن الحادث، متضمناً:
        - وصف الحادث بالتفصيل بناءً على المعلومات المتاحة.
        - تقييم نسبة الخطأ لكل طرف بناءً على البيانات المتوفرة (النسب المحتملة: [100%, 75%, 50%, 25%, 0%]).
        - تقييم الأضرار المادية لكل سيارة.

        يرجى عدم كتابة توصيات وموقع الحادث. وفي نهاية التقرير، اكتب أنه "قيد المراجعة".
        """
    )

    # Call the OpenAI API to generate the accident report based on the prompt
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "أنت مساعد في كتابة تقرير حادث مروري"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.1
    )

    # Return the generated report
    return response.choices[0].message['content'].strip()

    
# Button to generate accident report
if st.button("كتابة تقرير الحادث"):
    if (
        vehicle_reg_image1_file is not None and
        vehicle_reg_image2_file is not None
    ):
        # Ensure AccidentDescription is defined
        if 'AccidentDescription' not in locals():
            st.error("يرجى تحميل صورة الحادث وانتظار التحليل.")
        else:
            # Call RAG-based accident report generation
            with st.spinner("جاري كتابة تقرير الحادث..."):
                try:
                    accident_report = generate_accident_report_with_fault(FirstPartyDescription, SecondPartyDescription, AccidentDescription, vehicle_reg_image1_file, vehicle_reg_image2_file)
                    # Display the accident report
                    st.subheader("تقرير الحادث:")
                    st.write(accident_report)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء كتابة التقرير: {e}")
    else:
        st.error("الرجاء تحميل جميع الصور.")
