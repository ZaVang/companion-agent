from typing import List
import numpy as np
from scipy.spatial.distance import cdist
from pydantic import BaseModel


class Retrieve(BaseModel):
    pass

def retrieve_top_k_embeddings(embeddings: np.ndarray, 
                              query_embedding: np.ndarray, 
                              topk: int,
                              metric: str) -> List[int]:
    """
    检索与查询嵌入最相似的顶部 k 个嵌入的索引。

    参数:
    embeddings: 一个嵌入矩阵，形状为 (n_samples, embedding_dim)
    query_embedding: 查询的嵌入，形状为 (embedding_dim,)
    topk: 返回的相似嵌入数量

    返回:
    top_indices: 相似度最高的嵌入的索引列表
    """
    # 计算所有嵌入与查询嵌入之间的余弦距离
    distances = cdist([query_embedding], embeddings, metric=metric)[0]
    
    # 获取相似度最高的顶部 k 个嵌入的索引
    top_indices = np.argsort(distances)[:topk]
    
    return top_indices.tolist()
