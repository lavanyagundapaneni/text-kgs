import os
import json
import re
import pandas as pd
import PyPDF2
from docx import Document
from dotenv import load_dotenv
import boto3
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Get credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

# Initialize the Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Define a function to call Amazon Bedrock using the Converse API
def call_bedrock(prompt):
    try:
        conversation = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]
        response = bedrock_client.converse(
            modelId='mistral.mistral-7b-instruct-v0:2',
            messages=conversation,
            inferenceConfig={"maxTokens": 8192, "temperature": 0.7, "topP": 0.7},
            additionalModelRequestFields={"top_k": 50}
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Read unstructured data from file
def read_unstructured_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path).to_string(index=False)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    elif file_path.endswith('.pdf'):
        return load_pdf(file_path)
    else:
        raise ValueError("Unsupported file format")

# Load DOCX file
def load_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Load PDF file
def load_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Data Discovery: Extract nodes and relationships from unstructured data
def discover_data_insights(data):
    prompt = (
        "Extract all nodes and relationships from the following unstructured data. "
        "Provide a clear list of nodes and relationships without any special characters. "
        "Avoid using any special symbols like ** or //."
        "\n\nUnstructured Data:\n" + data
        )
    insights = call_bedrock(prompt)
    return insights

# Graph Data Modeling: Generate Cypher queries from insights
def generate_cypher_queries(insights):
    prompt = (
        "Generate Cypher queries to create nodes and relationships without duplicate nodes and each query should consists of single property only from the following data insights. "
        "Do not use MATCH keyword in relationships creation only use CREATE keyword. "
        "Format the queries without any special symbols and ensure they can be executed directly in Neo4j. "
        "generate each query line by line to avoid syntax."
        "Only create nodes do not include any properties"
        "Data Insights:\n" + insights
    )
    queries = call_bedrock(prompt)
    queries = re.sub(r'\s+', ' ', queries.strip())
    
    # Split by Cypher keywords to ensure proper formatting
    queries = re.sub(r'(\b(CREATE|MERGE)\b)', r'\n\1', queries)
    
    return queries

# Function to clean and validate the generated Cypher queries
def clean_and_validate_query(query):
    # Remove unnecessary characters
    query = re.sub(r'[\\]', '', query.strip())
    
    # Basic validation for Cypher syntax
    valid_keywords = ['CREATE', 'MATCH', 'MERGE', 'RETURN', 'DETACH', 'DELETE', 'SET', 'WITH']
    first_word = query.split()[0].upper()
    if first_word in valid_keywords:
        return query
    else:
        raise ValueError(f"Invalid Cypher query: {query}")

# Connect to Neo4j
neo4j_uri = "bolt://localhost:7687"  # Adjust as necessary
neo4j_user = "neo4j"
neo4j_password = "anuradha"

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Data Ingestion: Upload Cypher queries to Neo4j
def execute_cypher_queries(driver, queries):
    with driver.session() as session:
        for query in queries.split('\n'):
            if query.strip():
                try:
                    cleaned_query = clean_and_validate_query(query)
                    session.run(cleaned_query)
                    print(f"Executed: {cleaned_query}")
                except ValueError as ve:
                    print(f"Skipping invalid query: {ve}")
                except Exception as e:
                    print(f"Error executing query: {query}\n{e}")

# Main function
def main(file_path):
    unstructured_data = read_unstructured_data(file_path)
    
    # Data Discovery
    insights = discover_data_insights(unstructured_data)
    print("Data Insights:", insights)
    
    # Graph Data Modeling
    cypher_queries = generate_cypher_queries(insights)
    print("Cypher Queries:", cypher_queries)
    
    # Data Ingestion
    execute_cypher_queries(driver, cypher_queries)

# Path to the unstructured data file
file_path = 'C:/Users/chint/Downloads/SOP DRAFT KAARUNYA.docx'  # Replace with your file path

# Execute the main function
if __name__ == "__main__":
    main(file_path)
