import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('healthcare_dataset.csv')
df.head(5)

df.info()

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Num of Days'] = (df['Discharge Date']-df['Date of Admission']).dt.days
df =df.drop(columns=['Name','Date of Admission','Discharge Date'],axis=1)

df.info()

df.describe().plot(kind='bar',color='g')

df.isnull().sum()

df.duplicated().sum()

df = df.drop_duplicates()

df.shape

df['Test Results'] = df['Test Results'].replace({'Normal':1,'Inconclusive':0,'Abnormal':2})

plt.figure(figsize=(10,8))
sns.histplot(x='Age',data=df,kde=True,bins=20,hue='Gender')
plt.title('The gender wise age')
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
sns.scatterplot(x='Test Results',y='Age',data=df,hue='Gender',marker='D')
plt.title('The scatterplot data reulsts')
plt.legend()
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='Set1')
plt.title('The heatmap data info')
plt.legend()
plt.show()

numerical_col = df.select_dtypes(include=np.number).columns

for i in numerical_col:
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    sns.boxplot(x=i,ax=ax[0],data=df,palette='flare')
    ax[0].set_title(f'{i} distribution')
    sns.histplot(data=df,x=i,kde=True,bins=20,ax=ax[1],color='r')
    ax[1].set_title(f'{i} normal distribution')
    plt.tight_layout()
    plt.show()

object_col = df.select_dtypes(include='object').columns
object_col

unique_vals_analysis = pd.DataFrame(columns=['column','unique_value','num_of_unique_values'])
unique_rows=[]
for cols in object_col:
    new_row={'column': cols, 'unique_value': df[cols].unique(), 'num_of_unique_values': df[cols].nunique()}
    unique_rows.append(new_row)
unique_vals_analysis = pd.DataFrame(unique_rows)
unique_vals_analysis.set_index('column',inplace=True)
unique_vals_analysis

categorical_cols = ['Gender', 'Blood Type', 'Medical Condition','Insurance Provider', 'Admission Type', 'Medication', 'Test Results']

gender_billing = df.groupby('Gender')['Billing Amount'].sum().reset_index()
gender_billing['Billing Amount (in Millions)'] = gender_billing['Billing Amount'] / 1_000_000
print(gender_billing[['Gender', 'Billing Amount (in Millions)']])

for i in categorical_cols:
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    value_counts_df = df[i].value_counts(sort='ascending').reset_index()
    value_counts_df.columns = [i, 'count']
    y_min = value_counts_df['count'].min()
    y_max = value_counts_df['count'].max()
    y_min = max(0, (y_min // 1_000) * 1_000)
    y_max = ((y_max // 1_000) + 1) * 1_000
    sns.barplot(data=value_counts_df, x=i, y='count', palette = 'flare', ax=ax[0])
    ax[0].set_ylim(y_min, y_max)
    ax[0].set_title(f'{i} by count', fontsize = 15)
    group = df.groupby([i], sort=False)['Billing Amount'].sum().reset_index()
    group = group.sort_values('Billing Amount', ascending=False)
    billing_order = group[i]
    sns.barplot(data=group , x=i, y='Billing Amount', ax=ax[1], palette = 'crest', order=billing_order)
    y_min = group['Billing Amount'].min()
    y_max = group['Billing Amount'].max()
    y_min_rounded = max(0, (y_min // 1_000_000) * 1_000_000)
    y_max_rounded = ((y_max // 1_000_000) + 1) * 1_000_000
    ax[1].set_ylim(y_min_rounded, y_max_rounded)
    ax[1].set_title(f'{i} by Billing Amount', fontsize =15)
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))
    plt.tight_layout()
    plt.show()

cols_without_mc = ['Gender', 'Blood Type','Insurance Provider', 'Admission Type', 'Medication', 'Test Results']

df_male = df[df['Gender']=='Male']
df_female = df[df['Gender']=='Female']

for i in cols_without_mc:
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    group = df_male.groupby([i, 'Medical Condition'], sort= True)['Billing Amount'].sum().reset_index()
    sns.barplot(data=group, x=i, y='Billing Amount', ax=ax[0], palette = 'mako', hue='Medical Condition')
    ax[0].set_title(f'{i} and Medical Condition \n by Billing Amount for Male patients')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))
    group_count = df_female.groupby([i, 'Medical Condition'], sort= True)['Billing Amount'].sum().reset_index()
    sns.barplot(data=group_count, x=i, y='Billing Amount', ax=ax[1], palette = 'rocket_r', hue='Medical Condition')
    ax[1].set_title(f'{i} and Medical Condition \n by Billing Amount for Female patients')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

y = df_encoded.iloc[:,-1]
y.values

X = df_encoded.iloc[:,:-1].values
y = df_encoded.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training data shape {X_train.shape}')
print(f'Testing data shape {X_test.shape}')

y_train

models = {
    "Logistic Regression":LogisticRegression(),
    "K Means": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}
results = {}
n = len(cols)
rows, cols_per_row = 2, 2
fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 8))
axes = axes.flatten()  
idx=0
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy_scores = round(accuracy_score(y_test, predictions),3)
    results[model_name] = {'accuracy_scores': accuracy_scores}
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True,  cmap=sns.light_palette("purple", as_cmap=True), linewidths=0.7, linecolor='white', fmt='d',
                xticklabels=['Abnormal \n Predicted', 'Inconclusive \n Predicted', 'Normal \n Predicted'],
                yticklabels=['Actual \n Abnormal', 'Actual \n Inconclusive', 'Actual \n Normal'], ax = axes[idx], vmin=0, vmax=2000)
    axes[idx].set_title(f'Confusion Matrix of {model_name}')
    print(f'\033[1mAccuracy score of {model_name} is {accuracy_scores}\033[1m \n')
    idx+=1
plt.subplots_adjust(wspace=0.25, hspace=0.5)
plt.show()

results_df = pd.DataFrame(results).T
plt.figure(figsize = (10,5))
ax = sns.barplot(data = results_df, x=results_df.index, y=results_df['accuracy_scores'], palette="flare")
ax.bar_label(ax.containers[0], fontsize=10);
ax.set_xlabel('ML Algorithms')
plt.title('Performance Comparisions \n of Classification Algorithms \n', fontsize=16)
plt.tight_layout()
plt.show()

!pip install pytesseract

import pandas as pd
!sudo apt install tesseract-ocr
!pip install pytesseract
!pip install Pillow

from PIL import Image
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_lab_data(image_path):
    
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)

        # Split the text into lines
        lines = text.split('\n')

        extracted_data = []
        # Simple regex patterns to find potential lab test lines
        # This is a basic example and might need adjustment based on the report format
        # Pattern looks for: Text (Test Name), Number/Value (Measured Value), Optional Range, Optional Unit
        # This pattern is highly dependent on the exact format of your reports.
        # You will likely need to refine this based on real examples.
        pattern = re.compile(r'^(.*?)[\s:]+(\d+\.?\d*|-|[\<\>]\s*\d+\.?\d*)[\s:]*(\d+\.?\d*\s*-\s*\d+\.?\d*)?[\s:]*([a-zA-Z%/]+)?$')


        for line in lines:
            match = pattern.match(line.strip())
            if match:
                test_name = match.group(1).strip()
                measured_value = match.group(2)
                normal_range = match.group(3) if match.group(3) else ''
                unit = match.group(4) if match.group(4) else ''

                # Basic filtering to avoid extracting header/footer or irrelevant lines
                # You'll need to customize this based on common words in your reports
                if len(test_name) > 3 and not any(word in test_name.lower() for word in ['patient', 'report', 'date', 'doctor']):
                    extracted_data.append({
                        'Test Name': test_name,
                        'Measured Value': measured_value.strip(),
                        'Normal Range': normal_range.strip(),
                        'Unit': unit.strip()
                    })

        return extracted_data

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


image_path = 'Screenshot 2025-05-25 033231.png'

lab_results = extract_lab_data(image_path)

if lab_results:
    for result in lab_results:
        print(f"Test Name: {result['Test Name']}")
        print(f"Measured Value: {result['Measured Value']}")
        print(f"Normal Range: {result['Normal Range']}")
        print(f"Unit: {result['Unit']}")
        print("-" * 20)
else:
    print("No lab data extracted.")


if lab_results:
    df_lab_results = pd.DataFrame(lab_results)
    print("\nExtracted Data DataFrame:")
df_lab_results

import pandas as pd
def parse_range(range_str):
    """Parses a string representing a normal range into a tuple of (min, max)."""
    if not range_str:
        return None, None
    parts = range_str.split('-')
    if len(parts) == 2:
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            return None, None
    
    if '<' in range_str:
        try:
            return None, float(range_str.replace('<', '').strip())
        except ValueError:
            return None, None
    if '>' in range_str:
        try:
            return float(range_str.replace('>', '').strip()), None
        except ValueError:
            return None, None
    return None, None

def categorize_value(measured_value_str, normal_range_str):
    """Categorizes a measured value based on the normal range."""
    if not measured_value_str or not normal_range_str:
        return 'Unknown'

    try:
        measured_value = float(measured_value_str)
    except ValueError:
        return 'Unknown' 

    min_range, max_range = parse_range(normal_range_str)

    if min_range is not None and max_range is not None:
        if measured_value < min_range or measured_value > max_range:
           
            return 'Abnormal'
        else:
            return 'Normal'
    elif min_range is not None:
        if measured_value < min_range:
            return 'Abnormal'
        else:
            return 'Normal'
    elif max_range is not None:
        if measured_value > max_range:
            return 'Abnormal'
        else:
            return 'Normal'
    else:
        return 'Unknown' 

def structure_and_analyze_lab_data(extracted_lab_data):
    
    if not extracted_lab_data:
        return pd.DataFrame(columns=['Test Name', 'Measured Value', 'Normal Range', 'Unit', 'Range Status', 'Category'])

    df_lab_results = pd.DataFrame(extracted_lab_data)

    # Analyze range status and categorize
    df_lab_results['Range Status'] = df_lab_results.apply(
        lambda row: 'Outside Normal Range' if categorize_value(row['Measured Value'], row['Normal Range']) == 'Abnormal' else 'Within Normal Range',
        axis=1
    )
    df_lab_results['Category'] = df_lab_results.apply(
        lambda row: categorize_value(row['Measured Value'], row['Normal Range']),
        axis=1
    )

    return df_lab_results


if 'lab_results' in locals() and lab_results:
    df_analyzed_lab_results = structure_and_analyze_lab_data(lab_results)
    print("\nStructured and Analyzed Lab Data:")
    print(df_analyzed_lab_results)
elif 'df_lab_results' in locals() and not df_lab_results.empty:
     # If lab_results was not assigned, but df_lab_results exists from the previous step
    lab_results_list = df_lab_results.to_dict('records')
    df_analyzed_lab_results = structure_and_analyze_lab_data(lab_results_list)
    print("\nStructured and Analyzed Lab Data:")
    print(df_analyzed_lab_results)

else:
    print("\nNo lab data available for structuring and analysis. Please run extract_lab_data first.")

!python -m venv venv
!source venv/bin/activate  
!python -m venv venv
!source venv/bin/activate 
!pip install opencv-python pytesseract easyocr python-dotenv streamlit PyPDF2 pillow pdf2image
!pip install openai google-generativeai
!pip install openai google-generativeai

import google.generativeai as genai
import os

try:
    with open('code.txt', 'r') as f:
        api_key = f.read().strip()
    genai.configure(api_key=api_key)
except FileNotFoundError:
    print("Error: 'code.txt' not found. Please create this file and put your Gemini API key inside.")
except Exception as e:
    print(f"An error occurred while reading the API key: {e}")

def explain_lab_result(test_name, measured_value, normal_range, unit, category):
    
    if category == 'Normal' or category == 'Unknown':
        return f"The {test_name} result of {measured_value} {unit} is {category} and within the normal range ({normal_range})."

    prompt = f"""Explain in simple language what it means if a patient's {test_name} is {measured_value} {unit}, given the normal range is {normal_range}. This result is categorized as {category}. Please explain the potential implications in a concise manner."""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        # Access the text from the response
        explanation = response.text.strip()
        return explanation
    except Exception as e:
        return f"Could not generate explanation for {test_name}: {e}"

if 'df_analyzed_lab_results' in locals() and not df_analyzed_lab_results.empty:
    print("\nGenerative AI Explanations for Abnormal Results:")
    for index, row in df_analyzed_lab_results.iterrows():
        if row['Category'] == 'Abnormal':
            explanation = explain_lab_result(
                row['Test Name'],
                row['Measured Value'],
                row['Normal Range'],
                row['Unit'],
                row['Category']
            )
            print(f"\nExplanation for {row['Test Name']} ({row['Measured Value']} {row['Unit']}):")
            print(explanation)
else:
    
    print("\nNo analyzed lab data found with abnormal results to explain.")

import pandas as pd

!pip install nltk
import nltk
nltk.download("punkt_tab")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')


try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' resource found.")
except LookupError: # Catch LookupError which is raised when a resource is not found
    print("NLTK 'punkt' resource not found, downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' resource downloaded.")
except Exception as e:
    print(f"An unexpected error occurred while checking/downloading 'punkt': {e}")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("NLTK 'averaged_perceptron_tagger' resource found.")
except LookupError: # Catch LookupError
    print("NLTK 'averaged_perceptron_tagger' resource not found, downloading...")
    nltk.download('averaged_perceptron_tagger')
    print("NLTK 'averaged_perceptron_tagger' resource downloaded.")
except Exception as e:
    print(f"An unexpected error occurred while checking/downloading 'averaged_perceptron_tagger': {e}")

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
    print("NLTK 'maxent_ne_chunker' resource found.")
except LookupError: # Catch LookupError
    print("NLTK 'maxent_ne_chunker' resource not found, downloading...")
    nltk.download('maxent_ne_chunker')
    print("NLTK 'maxent_ne_chunker' resource downloaded.")
except Exception as e:
    print(f"An unexpected error occurred while checking/downloading 'maxent_ne_chunker': {e}")

try:
    nltk.data.find('corpora/words')
    print("NLTK 'words' resource found.")
except LookupError: # Catch LookupError
    print("NLTK 'words' resource not found, downloading...")
    nltk.download('words')
    print("NLTK 'words' resource downloaded.")
except Exception as e:
    print(f"An unexpected error occurred while checking/downloading 'words': {e}")


from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

def perform_basic_nlp(text):
    
    if not text:
        return {"tokens": [], "pos_tags": [], "named_entities": []}

   
    tokens = word_tokenize(text)

    
    pos_tags = pos_tag(tokens)

    
    named_entities = ne_chunk(pos_tags)

    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "named_entities": named_entities
    }


raw_image_text = ""
try:
    img_for_nlp = Image.open(image_path)
    raw_image_text = pytesseract.image_to_string(img_for_nlp)
    print("\n--- Raw Text Extracted for NLP ---")
    print(raw_image_text)
    print("----------------------------------")
    if not raw_image_text:
        print("\nWarning: No text was extracted for NLP processing.")
except FileNotFoundError:
    print(f"\nError: Image file not found at {image_path} for NLP processing.")
    raw_image_text = ""
except Exception as e:
    print(f"\nAn error occurred during text extraction for NLP: {e}")
    raw_image_text = ""

if raw_image_text:
    nlp_results = perform_basic_nlp(raw_image_text)

    print("\n--- Basic NLP Results ---")
    print("Tokens:", nlp_results["tokens"][:50], "...") # Print first 50 tokens
    print("\nPOS Tags:", nlp_results["pos_tags"][:50], "...") # Print first 50 POS tags
    print("\nNamed Entities (Tree structure):")
    
    print(nlp_results["named_entities"])
    print("-------------------------")
else:
    print("\nNo raw text available for basic NLP processing.")



def refined_extract_lab_data_with_pos(image_path):
    
    extracted_data = []
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)

        if not text:
            return extracted_data

        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)


        i = 0
        while i < len(pos_tags) - 2:
         
            if pos_tags[i][1].startswith('N') and pos_tags[i+1][1] in ['CD', 'NN', 'JJ'] and pos_tags[i+2][1].startswith('N'):
                 test_name = pos_tags[i][0]
                 measured_value = pos_tags[i+1][0]
                 unit = pos_tags[i+2][0]

                 if len(test_name) > 2 and not any(word in test_name.lower() for word in ['patient', 'report', 'date', 'doctor']):
                     extracted_data.append({
                         'Test Name': test_name,
                         'Measured Value': measured_value,
                         'Normal Range': 'Not Extracted by Basic NLP', # This method doesn't handle ranges well
                         'Unit': unit
                     })
                 i += 3 # Move past the potential pattern
            else:
                i += 1

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

!pip install opencv-python streamlit

import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, 0)  # grayscale
    denoised = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    _, binarized = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
    return binarized

!pip install pytesseract easyocr

import pytesseract
from PIL import Image

def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text
from PIL import Image

def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

import re

def structure_text(text):
    pattern = r'(?P<test>[A-Za-z ]+): (?P<value>[\d.]+) (?P<unit>\w+/[dDL]+)? \(Normal: (?P<range>[\d.-]+)\)'
    matches = re.findall(pattern, text)
    structured = [{"Test": m[0], "Value": m[1], "Unit": m[2], "Normal Range": m[3]} for m in matches]
    return structured

!pip install openai

import openai

openai.api_key = "YOUR_API_KEY"

def explain_result(test_name, value, normal_range):
    prompt = f"Explain in simple language what it means if the patient's {test_name} is {value}, given the normal range is {normal_range}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

openai.api_key = "YOUR_API_KEY"

def explain_result(test_name, value, normal_range):
    prompt = f"Explain in simple language what it means if the patient's {test_name} is {value}, given the normal range is {normal_range}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

def generate_summary(structured_data):
    summary = "Based on your test results, here are our findings:\n"
    for item in structured_data:
        if float(item["Value"]) < float(item["Normal Range"].split('-')[0]) or float(item["Value"]) > float(item["Normal Range"].split('-')[1]):
            summary += f"- {item['Test']} is out of normal range.\n"
    summary += "Please consult a physician for interpretation."
    return summary

!pip install streamlit

import streamlit as st

st.title("Medical Report Analyzer")

uploaded_file = st.file_uploader("Upload a medical report", type=["jpg", "png", "pdf"])
if uploaded_file:
    st.image(uploaded_file)
    
st.title("Medical Report Analyzer")

uploaded_file = st.file_uploader("Upload a medical report", type=["jpg", "png", "pdf"])
if uploaded_file:
    st.image(uploaded_file)
   

%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image
import pytesseract
import re
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import cv2
import os

warnings.filterwarnings('ignore')


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')



try:
    df = pd.read_csv('healthcare_dataset.csv')
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Num of Days'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
    df = df.drop(columns=['Name', 'Date of Admission', 'Discharge Date'], axis=1)
    df = df.drop_duplicates()
    
    label_encoders = {}
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

except FileNotFoundError:
    st.error("healthcare_dataset.csv not found. Please make sure it's uploaded or the path is correct.")
    df = None # Set df to None if file not found
except Exception as e:
    st.error(f"Error loading or preprocessing healthcare dataset: {e}")
    df = None

def extract_lab_data(image):
    
    try:
        text = pytesseract.image_to_string(image)
        lines = text.split('\n')
        extracted_data = []
        pattern = re.compile(r'^\s*(.*?)\s*[:\s]\s*([\d\.\<\>\-]+)\s*([a-zA-Z%/\(\)]+)?\s*\(?(?:Normal|Ref Range|Range)?\s*[:\s]?([\d\.\<\>\-\s]+)?\)?$')

        for line in lines:
            match = pattern.match(line.strip())
            if match:
                test_name = match.group(1).strip()
                measured_value = match.group(2).strip()
                unit = match.group(3).strip() if match.group(3) else ''
                normal_range = match.group(4).strip() if match.group(4) else ''

                if len(test_name) > 2 and not any(word in test_name.lower() for word in ['patient', 'report', 'date', 'doctor', 'result', 'test', 'unit']):
                    if measured_value and not any(char.isalpha() for char in measured_value if char not in ['<', '>','-','.']):
                        extracted_data.append({
                            'Test Name': test_name,
                            'Measured Value': measured_value,
                            'Normal Range': normal_range,
                            'Unit': unit
                        })

        return extracted_data, text

    except Exception as e:
        st.error(f"An error occurred during OCR extraction: {e}")
        return [], ""

def parse_range(range_str):
    if not range_str:
        return None, None
    parts = range_str.replace(' ', '').split('-')
    if len(parts) == 2:
        try:
            min_val = float(parts[0]) if parts[0] else None
            max_val = float(parts[1]) if parts[1] else None
            return min_val, max_val
        except ValueError:
            return None, None
    if '<' in range_str:
        try:
            return None, float(range_str.replace('<', ''))
        except ValueError:
            return None, None
    if '>' in range_str:
        try:
            return float(range_str.replace('>', '')), None
        except ValueError:
            return None, None
    try:
        single_value = float(range_str)
        return single_value, single_value 
    except ValueError:
         return None, None


def categorize_value(measured_value_str, normal_range_str):
    """Categorizes a measured value based on the normal range."""
    if not measured_value_str or not normal_range_str:
        return 'Unknown'

    cleaned_value_str = measured_value_str.replace('<', '').replace('>', '').strip()
    try:
        measured_value = float(cleaned_value_str)
    except ValueError:
        return 'Unknown' # Cannot compare non-numeric values

    min_range, max_range = parse_range(normal_range_str)

    if min_range is not None and max_range is not None:
        if measured_value < min_range or measured_value > max_range:
            return 'Abnormal'
        else:
            return 'Normal'
    elif min_range is not None: # Only minimum specified (e.g., > X)
        if measured_value < min_range:
            return 'Abnormal'
        else:
            return 'Normal'
    elif max_range is not None: # Only maximum specified (e.g., < X)
        if measured_value > max_range:
            return 'Abnormal'
        else:
            return 'Normal'
    else:
        return 'Unknown'

def structure_and_analyze_lab_data(extracted_lab_data):
    
    if not extracted_lab_data:
        return pd.DataFrame(columns=['Test Name', 'Measured Value', 'Normal Range', 'Unit', 'Range Status', 'Category'])

    df_lab_results = pd.DataFrame(extracted_lab_data)

    df_lab_results['Category'] = df_lab_results.apply(
        lambda row: categorize_value(row['Measured Value'], row['Normal Range']),
        axis=1
    )
    df_lab_results['Range Status'] = df_lab_results['Category'].apply(
        lambda cat: 'Outside Normal Range' if cat == 'Abnormal' else ('Within Normal Range' if cat == 'Normal' else 'Unknown')
    )


    return df_lab_results

def explain_lab_result(test_name, measured_value, normal_range, unit, category):
    
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    if not api_key:
        return "Gemini API key not configured."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        return f"Error configuring Gemini API: {e}"


    if category == 'Normal':
        return f"The {test_name} result of {measured_value} {unit} is normal and within the typical range ({normal_range})."
    elif category == 'Unknown':
        return f"The {test_name} result is {measured_value} {unit}. The normal range ({normal_range}) could not be fully processed, so its significance is unclear without further context."
    else: # Abnormal or other categories
        prompt = f"""Explain in simple language what it means if a patient's {test_name} is {measured_value} {unit}, given the normal range is {normal_range}. This result is categorized as {category}. Please explain the potential implications in a concise and easy-to-understand manner for a non-medical person."""

        try:
            response = model.generate_content(prompt)
            explanation = response.text.strip()
            return explanation
        except Exception as e:
            return f"Could not generate explanation for {test_name}: {e}"

st.title("Medical Data and Report Analyzer")

st.header("Analyze Medical Report Image")
uploaded_file = st.file_uploader("Upload a scanned medical report image (JPG, PNG)", type=["jpg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Report', use_column_width=True)

        st.write("Extracting data from the image...")

        extracted_lab_data, raw_text = extract_lab_data(image)

        if extracted_lab_data:
            st.subheader("Extracted Lab Data")
            df_extracted = pd.DataFrame(extracted_lab_data)
            st.dataframe(df_extracted)
            df_analyzed_lab_results = structure_and_analyze_lab_data(extracted_lab_data)

            if not df_analyzed_lab_results.empty:
                st.subheader("Analyzed Lab Data")
                st.dataframe(df_analyzed_lab_results)

                st.subheader("Explanation of Abnormal Results (using AI)")
                abnormal_results = df_analyzed_lab_results[df_analyzed_lab_results['Category'] == 'Abnormal']

                if not abnormal_results.empty:
                    for index, row in abnormal_results.iterrows():
                        explanation = explain_lab_result(
                            row['Test Name'],
                            row['Measured Value'],
                            row['Normal Range'],
                            row['Unit'],
                            row['Category']
                        )
                        st.write(f"**{row['Test Name']} ({row['Measured Value']} {row['Unit']})** - {row['Range Status']}")
                        st.info(explanation)
                else:
                    st.write("No abnormal results found in the extracted data.")

            st.subheader("Raw Text from Image")
            st.text(raw_text)

        else:
            st.warning("Could not extract structured lab data from the image. Please ensure the image is clear and the report format is standard.")
            st.subheader("Raw Text from Image")
            st.text(raw_text)


    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

st.header("Explore Healthcare Dataset (Pre-loaded)")

if df is not None: # Check if df was loaded successfully
    st.write("This section shows insights from a pre-loaded healthcare dataset.")

    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Healthcare Data")
        st.dataframe(df.head())

    st.subheader("Data Information")
    st.write(df.info()) 
    st.subheader("Data Statistics")
    st.write(df.describe())

    st.subheader("Age Distribution by Gender")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='Age', kde=True, bins=20, hue='Gender', ax=ax)
    ax.set_title('The gender wise age distribution')
    st.pyplot(fig)

    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='Set1', ax=ax)
        ax.set_title('Correlation Heatmap of Numeric Features')
        st.pyplot(fig)
    else:
        st.write("No numeric columns to display heatmap.")

else:
     st.warning("Healthcare dataset is not available for exploration.")
