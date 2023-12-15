import openai
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter    #for splitting the text to send it to LLM's
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

DOCUMENT_PATH = "docs/budget_speech.pdf"
EMBEDDING_EXPORT_FOLDER = "embeddings"
EXMBEDDING_EXPORT_FILE_NAME = "transformer_embeddings.csv"

loader = PyPDFLoader(DOCUMENT_PATH)
#Load PDF using pypdf into array of documents
#where each document contains the page content and metadata with page number.
pages  = loader.load_and_split()

#calculating embeddings
MODEL = "text-embedding-ada-002"  #OPENAI's best embedding model
BATCH_SIZE = 300    #valid upto 2048 embeddings input per request.
text_splitter = NLTKTextSplitter(chunk_size = 1000)

embeddings = []

for start in range(0,len(pages),BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = [x.page_content for x in pages[start:end]]  #from start(incl) to end(excl)
    batches = []
    for t in batch:
        batches.extend(text_splitter.split_text)    #naively split the large input into a bunch of smaller ones.
    batch = batches
    print(f"Batch {start} to {end - 1}")
    response = openai.Embedding.create(input = batch,model = MODEL)
    """
    Structure of a general response
    {
        "data": [
        {
            "embedding": [
                -0.0108,
                -0.0107,
                0.0323,
                ...
                -0.0114
            ],
            "index": 0, #this changes for the batch we process with i
            "object": "embedding"
        }
        ],
        "model": "text-embedding-ada-002",
        "object": "list"
    }
    In this we require "embedding"
    """
    print(response['data'])
    for i,j in enumerate(response['data']):
        assert i == j['index']       #checking embeddings are matching the order of the input
    batch_embeddings = response['data'][0]['embedding']
    embeddings.extend(batch_embeddings)

batch = [x.page_content for x in pages]
batches = []

for t in batch:
    batches.extend(text_splitter.split_text(t))
batch = batches
df = pd.DataFrame({"text":batch,"embeddings":embeddings})

#save document and chunks
SAVE_PATH = os.path.join(EMBEDDING_EXPORT_FOLDER,EXMBEDDING_EXPORT_FILE_NAME)
df.to_csv(SAVE_PATH,index = False)
