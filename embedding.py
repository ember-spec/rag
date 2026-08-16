# -*- coding: utf-8 -*-
"""阿里云通义 text-embedding-v3 自定义 LangChain Embedding.

通过 DashScope 的 OpenAI 兼容接口调用, 供离线入库脚本与后端服务共用。
"""
import time
from typing import List

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class AliDashScopeEmbeddings(Embeddings):
    """通义 text-embedding-v3 Embedding(OpenAI 兼容接口封装).

    - text-embedding-v3 单次请求最多 10 条文本, 内部自动分批
    - 内置指数退避重试, 应对限流/网络抖动
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v3",
        batch_size: int = 10,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\x00", " ").strip() or " " for t in texts[i : i + self.batch_size]]
            last_err: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    resp = self._client.embeddings.create(model=self.model, input=batch)
                    # 兼容接口按输入顺序返回
                    ordered = sorted(resp.data, key=lambda d: d.index)
                    vectors.extend(item.embedding for item in ordered)
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001 统一重试
                    last_err = e
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)
            if last_err is not None:
                raise RuntimeError(
                    f"调用 {self.model} 向量化失败(第 {i // self.batch_size + 1} 批): {last_err}"
                ) from last_err
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """入库场景: 批量向量化知识库切片文本。"""
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        """检索场景: 向量化用户查询。"""
        return self._embed_batch([text])[0]
