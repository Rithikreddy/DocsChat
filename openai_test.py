import pandas as pd
import numpy as np
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
import nltk
import ast

nltk.download('punkt')

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#nltk.download('punkt')
EMBEDDING_EXPORT_FOLDER = 'embeddings'

EMBEDDING_MODEL = 'text-embedding-ada-002'

#understand my resume and give me some suggestions
DOCUMENT_PATH = "Doc-Chat/transformer.pdf"

EMBEDDING_EXPORT_FILE_NAME = f'{DOCUMENT_PATH.split("/")[1].split(".")[0]}.csv'

loader = PyPDFLoader(DOCUMENT_PATH)
print(f"loader is {loader}")
#Load Documents and split into chunks. Chunks are returned as Documents.
pages = loader.load_and_split()
print(f"pages is {pages}")
#calculating embeddings
MODEL = "text-embedding-ada-002"  #OPENAI's best embedding model
BATCH_SIZE = 100
embeddings = []
text_splitter = NLTKTextSplitter(chunk_size = 200)

print(f"no of pages is {len(pages)}")

for start in range(0,len(pages),BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = [x.page_content for x in pages[start:end]]  #page_content is string of page content in document
    #print(type(batch))  #list of strings
    batches = []
    for t in batch:
        batches.extend(text_splitter.split_text(t))    #split incoming text and return chunks
    batch = batches
    response = client.embeddings.create(model = EMBEDDING_MODEL,input = batch)
    #debugging
    # with open('response.txt', 'w') as f:
    #     f.write(str(response))
    #batch_embeddings = response.data[0].embedding
    #print(batch_embeddings)
    for item in response.data:
        embeddings.append(item.embedding)

batch = [x.page_content for x in pages]
batches = []

for t in batch:
    batches.extend(text_splitter.split_text(t))
batch = batches

print("batch length is ",len(batch))
print("embeddings length is ",len(embeddings))

df = pd.DataFrame({"text":batch,"embedding":embeddings})

#save document chunk and embeddings
SAVE_PATH = os.path.join(
                EMBEDDING_EXPORT_FOLDER, 
                EMBEDDING_EXPORT_FILE_NAME
            )
df.to_csv(SAVE_PATH,index = False)

df = pd.read_csv(SAVE_PATH)
print(df)
# print(df['embedding'])

# x = df['embedding'].apply(ast.literal_eval)
# print(x[0])
#why every embedding size is 1536 -> It's the embedding size of the openAI embedding model
# for item in x:
#     print(item)
# print(len(x))