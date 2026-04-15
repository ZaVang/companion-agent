from typing import Literal, List, Dict, TYPE_CHECKING
from openai import AzureOpenAI
from utils.template import DEFAULT_PROMPT
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import numpy as np

# Lazy imports to avoid dependency issues at module load
if TYPE_CHECKING:
    from text2vec import SentenceModel

# Try to import jieba and gensim, but make them optional
try:
    import jieba
    from gensim.models import KeyedVectors
    _jieba_available = True
except (ImportError, AttributeError):
    _jieba_available = False

GPT35_dict = {
    "api_version": "2023-03-15-preview",
    "azure_endpoint": "your_endpoint_here",
    "api_key": "your_api_key_here",
    "azure_deployment": "your_deployment_here"
}

GPT4_dict = {
    "api_version": "2023-12-01-preview",
    "azure_endpoint": "your_endpoint_here",
    "api_key": "your_api_key_here",
    "azure_deployment": "your_deployment_here"
}

model_mapping = {
    "GPT3.5": GPT35_dict,
    "GPT4": GPT4_dict
}

def get_response_from_openai(query: str, 
                             system_prompt: str=DEFAULT_PROMPT, 
                             history: List[Dict[str, str]]=[], 
                             top_p: float=0.9, 
                             temperature: float=0.9, 
                             model_name: Literal['GPT3.5', 'GPT4']='GPT3.5',
                             **kwargs) -> str:
    '''
    history is like:
    [
        {
            "role": "user", 
            "content": "Knock knock."
        },
        {
            "role": "assistant", 
            "content": "Who's there?"
        }
    ]
    '''
    client = AzureOpenAI(**model_mapping[model_name])
    messages= [{"role": "system", "content": system_prompt}] 
    if query is not None:
        messages += history + [{"role": "user", "content": query}]
    response = client.chat.completions.create(
                model="<ignored>",
                messages = messages,
                top_p = top_p,
                temperature = temperature
            )

    return response.choices[0].message.content # type: ignore


class BGEModel:
    def __init__(self, model_path) -> None:
        self.model = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.dirname(model_path)
        )

    def encode(self, query):            
        encoded_dict = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        input_ids = np.array(encoded_dict["input_ids"])
        attention_mask = np.array(encoded_dict["attention_mask"])
        token_type_ids = np.array(encoded_dict["token_type_ids"])
        model_output = self.model.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}
        )
        model_output = model_output[0][:, 0]
        l2_norm = np.linalg.norm(model_output, ord=2, axis=1, keepdims=True)
        normalized_embeddings_np = model_output / l2_norm
        model_output = np.array([normalized_embeddings_np[0]])
        return model_output
        
        
class Word2VecModel:
    def __init__(self, model_path) -> None:
        if not _jieba_available:
            raise ImportError("jieba or gensim not available, Word2VecModel cannot be used")
        self.model = KeyedVectors.load_word2vec_format(model_path+'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt', binary=False)
        self.stopwords = self.load_stopwords(model_path+'/stopwords.txt')
        
    def load_stopwords(self, filepath):
        keywords = []
        with open(filepath, "r") as file:
            for line in file.readlines():
                keywords.append(line.strip())
        return keywords
    
    def encode(self, query):
        query = [word for word in jieba.cut(query) if word not in self.stopwords]
        embed = np.zeros(100)
        for word in query:
            try:
                embed += self.model[word]
            except:
                continue
        return embed
        
        
import signal
import functools

class _EmbeddingLoadTimeout(Exception):
    """Raised when embedding model loading exceeds timeout."""
    pass

def _timeout_handler(signum, frame):
    raise _EmbeddingLoadTimeout("Embedding model loading timed out")

def get_embedding_model(model_name='text2vec', timeout: int = 5):
    #support bge-large-zh and text2vec
    assert model_name in ['text2vec', 'bge', 'keyword'], 'currently only support text2vec, bge and keyword model'

    # Apply signal-based timeout on Linux (SIGALRM only works in main thread)
    using_timeout = [False]
    old_handler = None
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        using_timeout[0] = True

    try:
        if model_name=='text2vec':
            # Lazy import to avoid torch dependency at module load
            from text2vec import SentenceModel
            model = SentenceModel('models/text2vec-base-chinese-paraphrase')
        elif model_name=='bge':
            model = BGEModel('models/DPR/bge-large/bge.onnx')
        elif model_name=='keyword':
            model = Word2VecModel('models/tencent_word2vec/tencent-ailab-embedding-zh-d100-v0.2.0-s')
        return model
    except _EmbeddingLoadTimeout:
        raise
    except Exception as e:
        raise
    finally:
        if using_timeout[0] and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
    