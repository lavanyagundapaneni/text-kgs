# Text-KGs
This project demonstrates how to transform unstructured data into a structured knowledge graph using Amazon Bedrock and Neo4j. The process involves reading various data formats, extracting nodes and relationships using a language model (LLM) from Amazon Bedrock, generating Cypher queries to create the graph in Neo4j, and then uploading the data to Neo4j.

## Knowledge Graph:
A knowledge graph is a structured way of representing knowledge, employing a graph-based data model. It organizes information into nodes and edges, where nodes symbolize entities or concepts, and edges represent the relationships between them. This approach allows for efficient storage, retrieval, and inference of interconnected data, enhancing capabilities for advanced search, analysis, and reasoning tasks.

## Steps Involved in Creating and Utilizing a Knowledge Graph to Enhance LLMs:
### 1. Connecting to Amazon Bedrock and Neo4j:
  - Establish a connection with Amazon Bedrock using the boto3 library by setting up credentials like AWS Access Key, Secret 
    Access Key, and Region. This connection enables the use of language models to extract nodes and relationships from 
    unstructured data. Simultaneously, connect to a Neo4j database using the GraphDatabase driver from the neo4j library, 
    allowing you to run Cypher queries to store and manage the structured knowledge graph.
### 2. Reading and Preprocessing Unstructured Data :
Load and preprocess data from various unstructured formats like CSV, TXT, DOCX, and PDF files. The data is read and converted into a consistent format (like plain text) for further processing. This involves using different methods for each file type, such as loading CSV files into a DataFrame, reading text files line by line, parsing DOCX files using the python-docx library, and extracting text from PDF files using PyPDF2.
### 3. Extracting Entities and Relationships Using LLMs
Use a language model hosted on Amazon Bedrock to extract meaningful entities and relationships from the unstructured data. This is achieved by creating a prompt that instructs the LLM to identify relevant entities and their relationships. The prompt is sent to the Mistral model via the Bedrock client, and the resulting insights (nodes and relationships) are used to build the knowledge graph.
4. Generating Cypher Queries for Neo4j
Translate the extracted entities and relationships into Cypher queries that can be executed in Neo4j. This involves creating a prompt for the LLM to generate well-structured Cypher queries, which are free from duplicates and special characters. The generated queries are then cleaned and validated to ensure they conform to Neo4j’s syntax and best practices, making them ready for execution.
5. Ingesting Data into Neo4j
Execute the validated Cypher queries in Neo4j to build the knowledge graph. The queries are run using the Neo4j connection, creating nodes and relationships in the database. Any errors or invalid queries are handled gracefully, ensuring that only valid data is ingested into the knowledge graph.
6. Using the Knowledge Graph to Enhance LLMs
Leverage the structured knowledge graph to enhance the capabilities of language models by providing them with precise, context-rich data. This involves querying the graph using Cypher to fetch relevant data, which is then used as input to an LLM (e.g., LLaMA via LangChain). The LLM generates more accurate and context-aware natural language answers based on the graph data, which are then formatted to provide clear and concise responses.




