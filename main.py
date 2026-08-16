#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI RAG 客服后端.

流程: 用户 query → AliDashScopeEmbeddings 向量化 → Chroma 召回 top-N 切片
     → 组装 RAG 提示词(限定仅基于知识库回答) → 大模型流式生成 → SSE 推送前端
"""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from chromadb.config import Settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_chroma import Chroma
from openai import AsyncOpenAI
from pydantic import BaseModel

from embedding import AliDashScopeEmbeddings
from import_csv_to_chroma import build_documents, build_text_documents, load_rows_from_text, xlsx_to_csv_text

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_pro_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
LLM_API_KEY = os.getenv("LLM_API_KEY", "") or DASHSCOPE_API_KEY
TOP_K = int(os.getenv("TOP_K", "4"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

NO_ANSWER_TEXT = "暂无相关信息。知识库中暂时没有与您问题相关的资料，建议换个问法或联系人工客服。"

# 文件上传入库配置
ALLOWED_UPLOAD_EXTS = {".txt", ".md", ".csv", ".xlsx"}
MAX_UPLOAD_MB = 5

SYSTEM_PROMPT = (
    "你是企业内部知识库客服助手，必须严格遵守以下规则：\n"
    "1. 只能依据【知识库资料】中的内容回答用户问题，严禁使用你自己的知识、经验或推理编造答案；\n"
    "2. 如果【知识库资料】不足以回答用户问题，必须且只能回复：暂无相关信息；\n"
    "3. 使用简体中文回答，语气专业友好，条理清晰，内容较多时可分点说明。"
)

app = FastAPI(title="知识库客服问答系统")

embeddings = AliDashScopeEmbeddings(
    api_key=DASHSCOPE_API_KEY, base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL
)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
    client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
)
llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


class ChatRequest(BaseModel):
    query: str


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    query = req.query.strip()

    async def gen():
        if not query:
            yield sse({"type": "delta", "content": "请输入您的问题。"})
            yield "data: [DONE]\n\n"
            return

        # 1. 向量检索召回 top-N 知识库切片(cosine 相关度过滤)
        try:
            results = await asyncio.to_thread(
                vectorstore.similarity_search_with_relevance_scores, query, k=TOP_K
            )
            results = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "message": f"知识库检索失败: {e}"})
            yield "data: [DONE]\n\n"
            return

        # 2. 无有效召回: 不调用大模型, 直接返回固定话术, 杜绝编造
        if not results:
            yield sse({"type": "delta", "content": NO_ANSWER_TEXT})
            yield "data: [DONE]\n\n"
            return

        yield sse(
            {
                "type": "sources",
                "sources": [
                    {"question": doc.metadata.get("question", ""), "score": round(score, 3)}
                    for doc, score in results
                ],
            }
        )

        # 3. 组装 RAG 提示词, 调用大模型流式生成并逐块推送
        contexts = "\n\n".join(f"【资料{i + 1}】{doc.page_content}" for i, (doc, _) in enumerate(results))
        try:
            stream = await llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"【知识库资料】\n{contexts}\n\n【用户问题】\n{query}"},
                ],
                stream=True,
                temperature=0.3,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield sse({"type": "delta", "content": delta})
        except Exception as e:  # noqa: BLE001
            yield sse({"type": "error", "message": f"大模型调用失败: {e}"})

        # 4. 结束标记
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """接收前端上传的 txt/md/csv/xlsx 文件, 解析切片后向量化写入 Chroma。

    - csv: 按 question/human_answers 结构解析为 QA 知识
    - xlsx: openpyxl 读取第一个 sheet 转为 csv 流程处理
    - txt/md: 整体按 chunk_size 切片
    - 同名文件重复上传时先删旧切片再写入, 实现覆盖更新
    """
    filename = file.filename or "untitled"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        raise HTTPException(400, f"仅支持 {'/'.join(sorted(ALLOWED_UPLOAD_EXTS))} 格式文件")
    raw = await file.read()
    if not raw.strip():
        raise HTTPException(400, "文件内容为空")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"文件大小超过 {MAX_UPLOAD_MB}MB 限制")

    if ext == ".xlsx":
        try:
            text = await asyncio.to_thread(xlsx_to_csv_text, raw)
        except Exception as e:  # noqa: BLE001 openpyxl 解析失败统一提示
            raise HTTPException(400, f"xlsx 解析失败, 请确认为有效的 Excel 文件: {e}")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="ignore")

    if ext in (".csv", ".xlsx"):
        rows = load_rows_from_text(text)
        if not rows:
            raise HTTPException(400, "表格中未解析到有效知识条目(第二列问题、第三列答案)")
        docs, ids = build_documents(rows, source=filename)
    else:
        docs, ids = build_text_documents(filename, text)
    if not docs:
        raise HTTPException(400, "未生成任何有效切片")

    def _write():
        # 同名文件覆盖更新: 先删除该来源的旧切片
        vectorstore._collection.delete(where={"source": filename})
        vectorstore.add_documents(docs, ids=ids)
        return vectorstore._collection.count()

    total = await asyncio.to_thread(_write)
    return {"success": True, "filename": filename, "added": len(docs), "total": total}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
