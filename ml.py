import pinecone
from sentence_transformers import SentenceTransformer
from flask import current_app


class ProblemFixService:
    def __init__(self, flask, device='cpu'):
        self.device = device
        self.pineconeService = PineconeService("extractive-question-answering")

    def get_similar_for(self, questionText, limit):
        return self._get_context(questionText, top_k=limit)

    def _get_retriever(self):
        return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=self.device)

    def _get_context(self, question, top_k):
        retrieverEncoder = self._get_retriever()
        index = self.pineconeService.get_index()
        xq = retrieverEncoder.encode([question]).tolist()
        xc = index.query(xq, top_k=top_k, include_metadata=True)

        return [self._transform_result(x) for x in xc["matches"]]

    def _transform_result(self, result):
        return {
            'ai_id': result["id"],
            'problem_title': result["metadata"]["Title"],
            'score': result["score"],
        }


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
