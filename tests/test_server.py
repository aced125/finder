from google_detail.server import PythonPredictor
import logging

logger = logging.getLogger(__name__)


def test_server():
    p = PythonPredictor(config={"dummy_data": True})
    payload = {
        "action": "infer",
        "query": "Who is the father of Arya Stark?",
        "config": {"top_k_retriever": 10, "top_k_finder": 5},
    }
    output = p.predict(payload)
    logger.info(f"Output: {output}")
    # assert output == 1
