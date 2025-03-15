# Import Libraries
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from datetime import datetime, timedelta
from tqdm import tqdm
from pypdf import PdfReader
from docx import Document
import openai
from openai import OpenAI
import numpy as np
import pandas as pd
import glob
import json
import csv
import sys
import io
import os

# Redis
import redis
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
# https://redis-py.readthedocs.io/en/stable/examples/search_vector_similarity_examples.html
# https://redis.io/docs/latest/develop/get-started/vector-database/

# Supress langchain warning messages
import logging
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("openai_key")
az_connection_str = os.getenv("az_connection_str")
aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("aws_secret_access_key")

# Initialize OpenAI Client
openai_client = OpenAI(api_key=openai_key)

# Initialize Redis Client
redis_client = redis.Redis(host="localhost", port=6379)

# Parameters
DOC_PREFIX = "doc:"                         # RediSearch Key Prefix for the Index
EMBEDDING_MODEL = "text-embedding-3-large"  # Embedding model
EMBEDDING_DIMENSIONS = 3072                 # Embedding dimensions
CHUNK_SIZE = 5000                           # Maximum number of characters in each chunk 
                                            # 5000 characters ~ 800 Words ~ 1000 Tokens
CHUNK_OVERLAP = 200                         # Overlap between chunks

# Helper Functions
def get_embedding(text):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input = [text], model = EMBEDDING_MODEL).data[0].embedding

def split_text(doc_text):

    try:
    
        # Create an instance of CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            chunk_size    = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP
        )
        # Split the text
        chunks = text_splitter.split_text(doc_text)

    except Exception as e:
        print(f"Error while splitting text: {str(e)}")
        return []
    
    # Display the resulting chunks
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}:\n{chunk}\n")

    return chunks

def split_json(doc_json):
    
    try:

        # Create an instance of RecursiveJsonSplitter
        json_splitter = RecursiveJsonSplitter(
            max_chunk_size = CHUNK_SIZE
        )
        # Split the JSON
        chunks = json_splitter.split_json(json_data=doc_json)

        # Convert chunk from dict to string
        nchunks = []
        for chunk in chunks:
            nchunks.append(json.dumps(chunk, ensure_ascii=False)) 

    except Exception as e:
        print(f"Error while splitting JSON: {str(e)}")
        return []

    # Display the resulting chunks
    # for i, chunk in enumerate(nchunks):
    #     print(f"Chunk {i+1}:\n{chunk}\n")

    return nchunks

def create_index(index_name):
    try:
        # check to see if index exists
        redis_client.ft(index_name).info()
        print("Index already exists!")
    except:
        # schema
        schema = (
            TextField("index_name"),               # Text Field
            TextField("file_name"),                # Text Field
            TextField("chunk"),                    # Text Field
            TextField("content"),                  # Text Field
            TextField("source"),                   # Text Field
            TagField("tag"),                       # Tag Field Name
            VectorField("vector",                  # Vector Field Name
                "FLAT", {                          # Vector Index Type: FLAT or HNSW
                    "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                    "DIM": EMBEDDING_DIMENSIONS,   # Number of Vector Dimensions
                    "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                }
            ),
        )

        # index Definition
        definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
        #definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.JSON)

        # create Index
        redis_client.ft(index_name).create_index(fields=schema, definition=definition)
        print(f"Index {index_name} created!")

def insert_index(filename, source, doc_chunks, index_name):
    #print(filename, source, index_name)
    try:
        # create embeddings
        chunks_vec = []
        for chunk in doc_chunks:
            chunk_vector = np.array(get_embedding(chunk), dtype=np.float32).tobytes()
            chunks_vec.append(chunk_vector)
        # print(len(chunks_vec))
        
        # Write to Redis
        pipe = redis_client.pipeline()
        for i, sent_vec in enumerate(chunks_vec):
            unique_id = f"doc:{index_name}:{filename}:chunk_{i+1}"
            data = {
                "index_name": index_name,
                "file_name": filename,
                "chunk": "chunk_"+str(i+1),
                "content": doc_chunks[i],
                "source": source,
                "tag": "OpenAIEmbeddings",
                "vector": sent_vec,
            }
            pipe.hset(unique_id, mapping=data)
            
        res = pipe.execute()
        # print(f"Records inserted!")
    except Exception as e:
        print(str(e))

def read_txt(file_path, local=True):
    '''
    Reads Text file
    '''
    doc_chunks = []
    try:
        # Extract text
        if local:
            with open(file_path, "r") as file:
                doc_text = file.read()
        else:
            doc_text = file_path.decode('utf-8')

        # Split Text
        doc_chunks = split_text(doc_text)
        # print(f"No. of Chunks: {len(doc_chunks)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    return doc_chunks

def read_pdf(file_path):
    '''
    Reads PDF file
    '''
    doc_chunks = []
    try:
        # Extract text
        reader = PdfReader(file_path)
        doc_text = "\n".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
        # Split Text
        doc_chunks = split_text(doc_text)
        # print(f"No. of Chunks: {len(doc_chunks)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    return doc_chunks

def read_docx(file_path):
    '''
    Reads DOCX file
    '''
    doc_chunks = []
    try:
        # Extract text
        doc = Document(file_path)
        doc_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        # Split Text
        doc_chunks = split_text(doc_text)
        # print(f"No. of Chunks: {len(doc_chunks)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    return doc_chunks

def read_json(file_path, local=True):
    '''
    Reads JSON file
    '''
    doc_chunks = []
    try:
        # Extract text
        if local:
            with open(file_path, 'r') as file:
                doc_text = json.load(file)
        else:
            doc_text = json.loads(file_path)
            
        # Split Text
        doc_chunks = split_json(doc_text)
        # print(f"No. of Chunks: {len(doc_chunks)}")
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}")
    return doc_chunks

def read_csv(file_path, local=True):
    '''
    Reads CSV file
    '''
    doc_chunks = []
    try:
        doc_dict = {}
        # Extract text
        if local:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                count = 1
                for row in reader:
                    doc_dict[f"row {count}"] = row
                    count = count + 1
        else:
            reader = csv.DictReader(file_path)
            count = 1
            for row in reader:
                doc_dict[f"row {count}"] = row
                count = count + 1

        # Split Text
        doc_chunks = split_json(doc_dict)
        # print(f"No. of Chunks: {len(doc_chunks)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    return doc_chunks

def read_xlsx(file_path):
    '''
    Reads XLSX file
    '''
    doc_chunks = []
    try:
        doc_dict = {}
        # Extract text
        df = pd.read_excel(file_path, engine='openpyxl')
        dict_data = df.to_dict(orient='records')
        count = 1
        for row in dict_data:
            doc_dict[f"row {count}"] = row
            count = count + 1

        # Split Text
        doc_chunks = split_json(doc_dict)
        # print(f"No. of Chunks: {len(doc_chunks)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    return doc_chunks

def index_local(src_directory, index_name):

    # Create Index
    create_index(index_name)

    # Insert Index
    files = glob.glob(f"{src_directory}/*.*")
    source = "Local"
    #print(files)

    for i in tqdm(range(len(files)), desc="Indexing Files "):
        _, filename = os.path.split(files[i])
        ext = filename.split(".")[-1]
        # print(f"\nFile: {filename}")
        
        doc_chunks = []

        # Extract text
        if ext == "txt":
            doc_chunks = read_txt(files[i])
            
        elif ext == "pdf":
            doc_chunks = read_pdf(files[i])
            
        elif ext == "docx":
            doc_chunks = read_docx(files[i])

        elif ext == "json":
            doc_chunks = read_json(files[i])

        elif ext == "csv":
            doc_chunks = read_csv(files[i])

        elif ext == "xlsx":
            doc_chunks = read_xlsx(files[i])
            
        # Insert text into Redis
        insert_index(filename, source, doc_chunks, index_name)

def index_azure(container_name, index_name):

    from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas

    blob_service_client = BlobServiceClient.from_connection_string(az_connection_str)

    # use the client to connect to the container
    container_client = blob_service_client.get_container_client(container_name)

    # Create Index
    create_index(index_name)

    # Insert Index
    source = "Azure"

    blobs = list(container_client.list_blobs())  # Convert to list

    for i in tqdm(range(len(blobs)), desc="Indexing Files "):
        filename = blobs[i].name

        ext = filename.split(".")[-1]
        # print(f"\nFile: {filename}")

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
        file_data = blob_client.download_blob().readall()
        
        doc_chunks = []

        # Extract text
        if ext == "txt":
            doc_chunks = read_txt(file_data, False)
        
        elif ext == "pdf": 
            doc_chunks = read_pdf(io.BytesIO(file_data))
        
        elif ext == "docx":
            doc_chunks = read_docx(io.BytesIO(file_data))

        elif ext == "json":
            doc_chunks = read_json(file_data, False)

        elif ext == "csv":
            csv_data = blob_client.download_blob().content_as_text()
            doc_chunks = read_csv(io.StringIO(csv_data), False)

        elif ext == "xlsx":
            doc_chunks = read_xlsx(io.BytesIO(file_data))

        # Insert text into Redis
        insert_index(filename, source, doc_chunks, index_name)

def index_aws(bucket_name, index_name):

    import boto3

    s3_client = boto3.client(
        's3',
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
    )

    # List files in Bucket
    result = s3_client.list_objects_v2(Bucket=bucket_name)

    if "Contents" not in result:
        print("The bucket is empty.")

    # Create Index
    create_index(index_name)

    # Insert Index
    source = "AWS"

    for i in tqdm(range(len(result["Contents"])), desc="Indexing Files "):

        filename = result["Contents"][i]["Key"]
        ext = filename.split(".")[-1]
        # print(f"\nFile: {filename}")

        response = s3_client.get_object(Bucket=bucket_name, Key=filename)
        file_data = response['Body'].read()
        
        doc_chunks = []

        # Extract text
        if ext == "txt":
            doc_chunks = read_txt(file_data, False)
        
        elif ext == "pdf":
            doc_chunks = read_pdf(io.BytesIO(file_data))
        
        elif ext == "docx":
            doc_chunks = read_docx(io.BytesIO(file_data))

        elif ext == "json":
            doc_chunks = read_json(file_data, False)

        elif ext == "csv":
            doc_chunks = read_csv(io.StringIO(file_data.decode("utf-8")), False)

        elif ext == "xlsx":
            doc_chunks = read_xlsx(io.BytesIO(file_data))

        # Insert text into Redis
        insert_index(filename, source, doc_chunks, index_name)

# print(sys.argv)
if len(sys.argv) != 4:
    print("Invalid number of parameter passed")
    exit()

src = sys.argv[1]
src_dir = sys.argv[2]
index_name = sys.argv[3]

if src == "local":
    print("Indexing local files ...")
    index_local(src_dir, index_name)
elif src == "azure":
    print("Indexing Azure Blob Container files ...")
    index_azure(src_dir, index_name)
elif src == "aws":
    print("Indexing AWS S3 files ...")
    index_aws(src_dir, index_name)
else:
    print("Invalid Source Passed")