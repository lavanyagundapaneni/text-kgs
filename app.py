import os
import json
import re
import pandas as pd
import PyPDF2
import networkx as nx
import matplotlib.pyplot as plt
from docx import Document
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to call ChatGPT for extracting nodes and relationships
def call_chatgpt(prompt):
    try:
        client = openai.OpenAI()  # Initialize the OpenAI client
        
        response = client.chat.completions.create(  # Updated API call
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are an expert in knowledge graph extraction."},
                      {"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.5,
            top_p=0.9
        )
        return response.choices[0].message.content  # Corrected response handling
    except Exception as e:
        print(f"Error accessing ChatGPT: {e}")
        raise

# Function to read different file types
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

# Repair and parse JSON from LLM response
def repair_and_parse_json(insights):
    try:
        start_index = insights.find('{')
        end_index = insights.rfind('}') + 1
        if start_index == -1 or end_index == -1:
            raise ValueError("No valid JSON object found in the response.")

        json_data = insights[start_index:end_index]
        
        try:
            from json_repair import repair_json
            repaired_json = repair_json(json_data)
            return json.loads(repaired_json)
        except ImportError:
            return json.loads(json_data)
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise ValueError("Invalid JSON format")

# Extract nodes and relationships using ChatGPT
def extract_nodes_and_relationships(data):
    prompt = (
        "You are an expert in extracting nodes and relationships frpom the given data. "
        "Extract all nodes and relationships from the following text and return them in valid JSON format.\n\n"
        "### Example:\n"
        "Input:\n"
        "Grade 9 math questions:\n"
        "- What is 2+2?\n"
        "- What is the Pythagorean theorem?\n"
        "Output:\n"
        '{\n'
        '    "nodes": [\n'
        '        {"id": "Grade 9", "type": "Grade"},\n'
        '        {"id": "Mathematics", "type": "Subject"},\n'
        '        {"id": "What is 2+2?", "type": "Question"},\n'
        '        {"id": "What is the Pythagorean theorem?", "type": "Question"}\n'
        '    ],\n'
        '    "relationships": [\n'
        '        {"source": "Grade 9", "target": "Mathematics", "type": "HAS_SUBJECT"},\n'
        '        {"source": "Mathematics", "target": "What is 2+2?", "type": "HAS_QUESTION"},\n'
        '        {"source": "Mathematics", "target": "What is the Pythagorean theorem?", "type": "HAS_QUESTION"}\n'
        '    ]\n'
        '}\n\n'
        "### Unstructured Data:\n" + data
    )
    insights = call_chatgpt(prompt)
    return insights

# Function to build a knowledge graph using NetworkX
def build_knowledge_graph(insights_json):
    G = nx.DiGraph()  

    if isinstance(insights_json, dict) and "relationships" in insights_json:
        relationships = insights_json["relationships"]
    elif isinstance(insights_json, list):
        relationships = [rel for rel in insights_json if "source" in rel and "target" in rel]
    else:
        raise ValueError("Unexpected format for insights_json")

    # Extract unique nodes
    nodes = set()
    for rel in relationships:
        nodes.add(rel["source"])
        nodes.add(rel["target"])

    # Add nodes to the graph
    for node in nodes:
        G.add_node(node)

    # Add relationships (edges)
    for rel in relationships:
        G.add_edge(rel["source"], rel["target"], type=rel["type"])

    return G

# Function to visualize the knowledge graph
def visualize_graph(G):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)

    edge_labels = {(u, v): G.edges[u, v]['type'] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Knowledge Graph")
    plt.show()

# Main function
def main(file_path):
    # Read unstructured data
    unstructured_data = read_unstructured_data(file_path)

    # Extract nodes and relationships
    raw_insights = extract_nodes_and_relationships(unstructured_data)

    print("Raw Data Insights:", raw_insights)  

    try:
        insights_json = repair_and_parse_json(raw_insights)
        print("JSON Insights:", json.dumps(insights_json, indent=4))

        # Build knowledge graph
        G = build_knowledge_graph(insights_json)

        # Visualize knowledge graph
        visualize_graph(G)

    except ValueError as e:
        print(f"Error processing data: {e}")

# File path to unstructured data (replace with your actual path)
file_path = "C:/Users/chint/Downloads/grade_subject_questions.csv"

if __name__ == "__main__":
    main(file_path)
