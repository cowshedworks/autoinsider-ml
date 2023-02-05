import pinecone
import mysql.connector
from sentence_transformers import SentenceTransformer
from flask import current_app
from sys import getsizeof
from pprint import pprint
import pandas as pd


class ProblemFixService:
    def __init__(self, device='cpu'):
        self.device = device
        self.pineconeService = PineconeService("extractive-question-answering")

    def get_similar_for(self, questionText, limit):
        return self._get_context(questionText, top_k=limit)

    def rebuild_index(self):
        self.pineconeService.rebuild_index()

    def add_to_index(self, df):
        retriever_encoder = self._get_retriever()
        index = self.pineconeService.get_index()

        # we will use batches of 64
        batch_size = 64

        for i in range(0, len(df), batch_size):
            # find end of batch
            i_end = min(i+batch_size, len(df))
            # extract batch
            batch = df.iloc[i:i_end]
            # generate embeddings for batch
            emb = retriever_encoder.encode(batch['Context'].tolist()).tolist()
            ids = batch['ID'].astype(str).tolist()
            meta = batch[['Title']].to_dict(orient="records")
            to_upsert = list(zip(ids, emb, meta))
            _ = index.upsert(vectors=to_upsert)

        return len(df)

    def _get_retriever(self):
        return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=self.device)

    def _get_context(self, question, top_k):
        retriever_encoder = self._get_retriever()
        index = self.pineconeService.get_index()
        xq = retriever_encoder.encode([question]).tolist()
        xc = index.query(xq, top_k=top_k, include_metadata=True)

        return [self._transform_result(x) for x in xc["matches"]]

    def _transform_result(self, result):
        return {
            'ai_id': result["id"],
            'problem_title': result["metadata"]["Title"],
            'score': result["score"],
        }


class MYSQLService:
    def __init__(self):
        self.connection = self.connect()

    def connect(self):
        return mysql.connector.connect(
            host=current_app.config["MYSQL_HOST"],
            user=current_app.config["MYSQL_USER"],
            password=current_app.config["MYSQL_PASSWORD"],
            database=current_app.config["MYSQL_DATABASE"],
            port=current_app.config["MYSQL_POST"]
        )

    def get_all_problems(self):
        mycursor = self.connection.cursor()
        mycursor.execute("""
            SELECT
                problems.id,
                problems.title,
                CONCAT(
                    problems.title,
                    ' ',
                    problems.problem
                ) AS context
            FROM problems
            JOIN cars
            ON cars.id = problems.car_id
            JOIN manufacturers
            ON manufacturers.id = cars.manufacturer_id
            WHERE problems.problem != ''
                AND problems.status = 1
                AND cars.id = problems.car_id
        """)
        myresult = mycursor.fetchall()

        return pd.DataFrame(myresult, columns=[
            "ID", "Title", "Context"])


class PineconeService:
    def __init__(self, index):
        self.index = index
        self.connect()

    def connect(self):
        pinecone.init(
            api_key=current_app.config["PINECONE_API_KEY"],
            environment=current_app.config["PINECONE_ENV"]
        )

    def get_index(self):
        return pinecone.Index(self.index)

    def rebuild_index(self):
        if self.index in pinecone.list_indexes():
            pinecone.delete_index(self.index)

        # create the index if it does not exist
        pinecone.create_index(
            self.index,
            dimension=384,
            metric="cosine"
        )
