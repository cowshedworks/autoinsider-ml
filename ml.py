import pinecone
from sentence_transformers import SentenceTransformer
from flask import current_app


class ProblemFixService:
    def __init__(self, flask, device='cpu'):
        self.device = device
        self.pineconeService = PineconeService("extractive-question-answering")

    def getSimilarFor(self, questionText, limit):
        return self._getContext(questionText, top_k=limit)

    def _getRetriever(self):
        return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=self.device)

    def _getContext(self, question, top_k):
        retrieverEncoder = self._getRetriever()
        index = self.pineconeService.getIndex()

        # generate embeddings for the question
        xq = retrieverEncoder.encode([question]).tolist()
        # search pinecone index for context passage with the answer
        xc = index.query(xq, top_k=top_k, include_metadata=True)
        # extract the context passage from pinecone search result
        c = [self._transformResult(x) for x in xc["matches"]]
        return c

    def _transformResult(self, result):
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

    def getIndex(self):
        return pinecone.Index(self.index)

    def rebuildIndex(self):
        if self.index in pinecone.list_indexes():
            pinecone.delete_index(self.index)

        # create the index if it does not exist
        pinecone.create_index(
            self.index,
            dimension=384,
            metric="cosine"
        )
