import pinecone
import mysql.connector
from sentence_transformers import SentenceTransformer
from flask import current_app
from sys import getsizeof
from pprint import pprint
import pandas as pd
from abc import ABC, abstractmethod


class BaseSimilarContentService(ABC):
    def __init__(self, device='cpu', pineconeIndexName='no-valid-index'):
        self.device = device
        self.pineconeService = PineconeService(pineconeIndexName)

    def get_similar_for(self, queryText, limit):
        return self._get_context(queryText, top_k=limit)

    def rebuild_index(self):
        self.pineconeService.rebuild_index()

    def delete_from_index(self, vector_ids):
        index = self.pineconeService.get_index()
        index.delete(ids=vector_ids)

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

    def _get_context(self, queryText, top_k):
        retriever_encoder = self._get_retriever()
        index = self.pineconeService.get_index()
        xq = retriever_encoder.encode([queryText]).tolist()
        xc = index.query(xq, top_k=top_k, include_metadata=True)

        return [self._transform_result(x) for x in xc["matches"]]

    @abstractmethod
    def _transform_result(self, result):
        pass


class AutoInsiderService(BaseSimilarContentService):
    def __init__(self):
        BaseSimilarContentService.__init__(
            self,
            pineconeIndexName='autoinsider-similar-problems'
        )

    def _transform_result(self, result):
        return {
            'ai_id': result["id"],
            'problem_title': result["metadata"]["Title"],
            'score': result["score"],
        }


class EuropeanRailGuideService(BaseSimilarContentService):
    def __init__(self):
        BaseSimilarContentService.__init__(
            self,
            pineconeIndexName='erg-similar-places'
        )

    def _transform_result(self, result):
        return {
            'erg_id': result["id"],
            'place_name': result["metadata"]["Title"],
            'score': result["score"],
        }


class MYSQLService:
    def __init__(self, host, user, password, database, port):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = self.connect()

    def connect(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port
        )

    def get_all_places(self):
        mycursor = self.connection.cursor()
        mycursor.execute("""
            SELECT
                must_visit.id,
                must_visit.name,
                CONCAT(
                    must_visit.name,
                    ' ',
                    must_visit.description
                ) AS context
            FROM must_visit
            WHERE must_visit.description != ''
                AND must_visit.active = 1
        """)
        myresult = mycursor.fetchall()

        return pd.DataFrame(myresult, columns=[
            "ID", "Title", "Context"])

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
