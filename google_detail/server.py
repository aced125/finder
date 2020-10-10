import torch
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack import Finder
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever, TfidfRetriever
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http


class PythonPredictor:
    def __init__(self, config):
        gpu = torch.cuda.is_available()
        print(f"GPU available? ", gpu)
        # self.reader = FARMReader(
        #     model_name_or_path="deepset/roberta-base-squad2", use_gpu=gpu
        # )
        self.reader = TransformersReader(
            model="elgeish/cs224n-squad2.0-albert-base-v2",
            tokenizer="elgeish/cs224n-squad2.0-albert-base-v2",
            context_window_size=70,
            use_gpu=0 if gpu else -1,
            top_k_per_candidate=4,
            return_no_answers=False,
            max_seq_len=256,
            doc_stride=128,
        )

        # self.doc_store = ElasticsearchDocumentStore(
        #     host="localhost", username="", password="", index="document"
        # )
        self.doc_store = InMemoryDocumentStore()

        if config.get("dummy_data"):
            # Initially populate the documents with Game of Thrones
            doc_dir = "data/article_txt_got"
            s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
            fetch_archive_from_http(url=s3_url, output_dir=doc_dir)
            dicts = convert_files_to_dicts(
                dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True
            )
            self.doc_store.write_documents(dicts)

        self.doc_store.write_documents(
            [
                {"name": "ww2", "text": "World war 2 ended in 1945."},
                {
                    "name": "google",
                    "text": "Google was started by 2 Stanford grads in a garage, and is now a 1 trillion dollar company.",
                },
                {
                    "name": "Amzn",
                    "text": "Amazon is a huge company built by Jeff Bezos.",
                },
            ]
        )

        # self.retriever = ElasticsearchRetriever(document_store=self.doc_store)
        self.retriever = TfidfRetriever(document_store=self.doc_store)
        self.finder = Finder(self.reader, self.retriever)

    def infer(self, payload):
        query = payload["query"]
        config = payload.get("config", {})
        prediction = self.finder.get_answers(
            query,
            top_k_retriever=config.get("top_k_retriever", 10),
            top_k_reader=config.get("top_k_reader", 5),
        )
        return prediction

    def list(self, payload):
        return [{"text": doc.text} for doc in self.doc_store.get_all_documents()]

    def store(self, payload):
        documents = payload["documents"]
        self.doc_store.write_documents(documents)
        self.retriever = TfidfRetriever(document_store=self.doc_store)
        self.finder = Finder(self.reader, self.retriever)
        return {"status_code": 200, "message": "success"}

    def predict(self, payload):
        action: str = payload["action"]
        action_types = ["infer", "store", "list"]
        assert action in action_types, f"`action` ({action}) must be in {action_types}"
        return getattr(self, action)(payload)
