import ast
import os
import pandas as pd
import numpy as np
import tiktoken
from scipy import spatial
import torch
import json
import replicate
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class DocumentChat:
    def __init__(
        self,
        EMBEDDING_PATH,
        INTRODUCTION_QUESTION,
        SYSTEM_CONTEXT_MESSAGE,
        DOCUMENT_NAME,
        APIKEY,
        EMBEDDING_MODEL="text-embedding-ada-002",
        GPT_MODEL="gpt-3.5-turbo"
    ):
        self.EMBEDDING_PATH = EMBEDDING_PATH
        self.INTRODUCTION_QUESTION = INTRODUCTION_QUESTION
        self.SYSTEM_CONTEXT_MESSAGE = SYSTEM_CONTEXT_MESSAGE
        self.DOCUMENT_NAME = DOCUMENT_NAME
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.GPT_MODEL = GPT_MODEL
        df = pd.read_csv(EMBEDDING_PATH)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        self.df = df

    def strings_ranked_by_relatednesses(
        self,
        query,
        df,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n=300
    ):
        query_embedding = []
        query_embedding_response = client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=query
        )
        query_embedding = query_embedding_response.data[0].embedding
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"])) for i, row in df.iterrows()
        ]
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_n], relatednesses[:top_n]

    def query_message(
        self,
        query,
        df,
        model,
        token_budget
    ):
        strings, relatednesses = self.strings_ranked_by_relatednesses(query, df)
        intro = f'{self.INTRODUCTION_QUESTION}'
        question = f"\n\nQuestion:{query}"
        message = intro
        for string in strings:
            next_article = f'\n\n {self.DOCUMENT_NAME}\n"""\n{string}\n"""'
            if (
                self.num_tokens(message + next_article + question, model=model) > token_budget
            ):
                break
            else:
                message += next_article
        return message + question

    def num_tokens(
        self,
        text,
        model
    ):
        model = self.GPT_MODEL
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def llama2(self, messages):
        res = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "debug": False,
                "top_k": 70,
                "top_p": 1,
                "prompt": messages[0]["content"],
                "system_prompt": messages[1]["content"]
            }
        )
        response_list = list(res)
        response_msg = ""
        for output in response_list:
            response_msg += output
        return response_msg

    def openai(self, messages):
        response = client.chat.completions.create(
            model=self.GPT_MODEL,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def retrieve(self, llm, messages):
        if llm == 'llama2':
            return self.llama2(messages)
        if llm == 'openai':
            return self.openai(messages)

    def ask(
        self,
        query,
        previous_qas,
        input_model,
        token_budget=4096 - 1500,
        print_message=False,
    ):
        model = self.GPT_MODEL
        df = self.df
        message = self.query_message(query, df, model=model, token_budget=token_budget)
        system_message = f"{self.SYSTEM_CONTEXT_MESSAGE}"
        for previous_qa in previous_qas:
            system_message += f"\nYFor this question {previous_qa.get('question')} this is the answer provided {previous_qa.get('answer')}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]

        return self.retrieve(input_model, messages)

if __name__ == "__main__":
    EMBEDDING_PATH = "embeddings/transformer.csv"
    INTRODUCTION_QUESTION = 'Use the below Research paper Documentation to answer the subsequent question. If the answer cannot be found in the article, write "I could not find an answer."'
    SYSTEM_CONTEXT_MESSAGE = "You answer questions about Research paper"
    DOCUMENT_NAME = "Research Paper Documentation :"
    API_KEY = os.environ["OPENAI_API_KEY"]
    docChat = DocumentChat(EMBEDDING_PATH, INTRODUCTION_QUESTION, SYSTEM_CONTEXT_MESSAGE, DOCUMENT_NAME, API_KEY)
    question = "what is multi head attention please summarize it in 5 sentences?"
    previous_qas = ""
    models = ['openai', 'llama2']
    messages = []
    for model in models:
        message = docChat.ask(query=question, previous_qas=previous_qas, print_message=True, input_model=model)
        messages.append(message)
    print("question is \n\n", question)
    for i in range(len(messages)):
        print("\n\n************************\n\n")
        print(f"response using the model {models[i]} is \n\n", messages[i])
