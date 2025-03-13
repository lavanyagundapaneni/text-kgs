import os
import pandas as pd
import numpy as np
import faiss
import openai
import json
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# ------------------ Step 1: Load & Chunk CSV Data ------------------
def load_and_chunk_csv(file_path, chunk_size=1000, overlap=100):
    """Reads CSV, converts to text, and splits into meaningful chunks while tracking word counts."""
    df = pd.read_csv(file_path)
    text_data = df.to_string(index=False)  # Convert entire CSV to text

    # Count total words in original data
    total_words_original = len(text_data.split())

    # Use LangChain's RecursiveCharacterTextSplitter for semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text_data)

    # Count total words in chunked data
    total_words_chunked = sum(len(chunk.split()) for chunk in chunks)

    print(f"Total words in original data: {total_words_original}")
    print(f"Total words in chunked data: {total_words_chunked}")
    print(f"Number of Chunks Created: {len(chunks)}")

    return chunks, total_words_original, total_words_chunked

# ------------------ Step 2: Store Embeddings in FAISS ------------------
def store_embeddings_faiss(chunks):
    """Generates embeddings and stores them in FAISS."""
    vector_store = FAISS.from_texts(chunks, embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

# ------------------ Step 3: Cosine Similarity Check ------------------
def calculate_cosine_similarity(chunks):
    """Checks cosine similarity between all chunks to ensure proper chunking."""
    chunk_embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
    chunk_matrix = np.array(chunk_embeddings)

    # Compute cosine similarity between chunks
    similarity_matrix = cosine_similarity(chunk_matrix)

    print(" Cosine Similarity Matrix (Sample):")
    print(similarity_matrix[:5, :5])  # Print first 5x5 values for inspection

    return similarity_matrix

# ------------------ Step 4: Extract Nodes & Relationships ------------------
def call_chatgpt(prompt):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Initialize with API key

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in knowledge graph extraction."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.5,
            top_p=0.9
        )

        # Extract response content
        raw_output = response.choices[0].message.content.strip()

        # Ensure we extract only JSON
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No valid JSON found in response.")

        json_string = raw_output[json_start:json_end]

        return json.loads(json_string)

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Raw API response: {raw_output}")
        return {"nodes": [], "relationships": []}

    except Exception as e:
        print(f"Error accessing ChatGPT: {e}")
        return {"nodes": [], "relationships": []}


def extract_nodes_relationships(chunks):
    """Extracts nodes and relationships from text using ChatGPT."""
    extracted_data = []

    for i, chunk in enumerate(chunks):
        prompt = (
            "You are an expert in extracting nodes and relationships from the given data. "
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
            "### Unstructured Data:\n" + chunk
        )

        extracted_json = call_chatgpt(prompt)

        # Check if valid extraction happened
        if not extracted_json or "nodes" not in extracted_json or "relationships" not in extracted_json:
            print(f"Skipping chunk {i+1} due to extraction issues.")
            extracted_json = {"nodes": [], "relationships": []}

        print(f"\nChunk {i+1}: Extracted Nodes & Relationships")
        print(json.dumps(extracted_json, indent=2))

        extracted_data.append(extracted_json)

    return extracted_data

# ------------------ Step 5: Build & Visualize Knowledge Graph ------------------
def build_knowledge_graph(extracted_data):
    """Creates a knowledge graph using NetworkX."""
    G = nx.DiGraph()  # Directed Graph

    for entry in extracted_data:
        nodes = entry.get("nodes", [])
        relationships = entry.get("relationships", [])

        # Add nodes (ensure they are correctly formatted)
        for node in nodes:
            node_id = node.get("id")  # Extract 'id' from the dictionary
            if node_id:  # Ensure node_id is not None
                G.add_node(node_id, label=node.get("type", "Entity"))

        # Add edges (relationships)
        for rel in relationships:
            source = rel.get("source")
            target = rel.get("target")
            relation = rel.get("type", "related_to")  # Ensure a default type

            if source and target:  # Ensure nodes exist before adding edges
                G.add_edge(source, target, label=relation)

    return G

def visualize_knowledge_graph(G):
    """Visualizes the knowledge graph using NetworkX and Matplotlib."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for visualization

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=3000, font_size=10, font_weight="bold")

    # Draw edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Knowledge Graph Visualization")
    plt.show()

# ------------------ Run the Pipeline ------------------
if __name__ == "__main__":
    csv_path = "C:/Users/chint/Downloads/grade_subject_questions.csv"

    # Step 1: Chunking & Word Count Validation
    chunks, original_words, chunked_words = load_and_chunk_csv(csv_path)

    # Step 2: Store Embeddings in FAISS
    vector_store = store_embeddings_faiss(chunks)

    # Step 3: Compute Cosine Similarity
    similarity_matrix = calculate_cosine_similarity(chunks)

    # Step 4: Extract Nodes & Relationships using LLM
    extracted_kg_data = extract_nodes_relationships(chunks)

    # Step 5: Build and Visualize Knowledge Graph
    G = build_knowledge_graph(extracted_kg_data)
    
    visualize_knowledge_graph(G)
    print("Nodes in the Knowledge Graph:")
print(G.nodes(data=True))

print("\nEdges in the Knowledge Graph:")
print(G.edges(data=True))
    
