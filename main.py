#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI RAG 客服后端.

流程: 用户 query → AliDashScopeEmbeddings 向量化 → Chroma 召回 top-N 切片
     → 组装 RAG 提示词(限定仅基于知识库回答) → 大模型流式生成 → SSE 推送前端
"""
import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("rag-pro")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    # uvicorn 启动会重配 root logger, 关闭传播避免日志被吞
    log.propagate = False

import uvicorn
from chromadb.config import Settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_chroma import Chroma
from openai import AsyncOpenAI
from pydantic import BaseModel

from archive_extractor import ArchiveError, extract_archive, is_zip
from embedding import AliDashScopeEmbeddings
from import_csv_to_chroma import (
    build_documents,
    build_text_documents,
    docx_to_text,
    load_rows_from_text,
    pdf_to_text,
    xlsx_to_csv_text,
)

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
ALLOWED_UPLOAD_EXTS = {".txt", ".md", ".csv", ".xlsx", ".docx", ".pdf", ".zip"}
MAX_UPLOAD_MB = 5          # 普通文档上传大小上限(MB)
MAX_ARCHIVE_MB = 20       # 压缩包上传大小上限(MB)

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


def _decode_text(raw: bytes) -> str:
    """尝试 utf-8-sig, 失败回退 gbk 解码纯文本。"""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="ignore")


async def _build_docs_for_file(filename: str, raw: bytes):
    """处理单个文档字节, 返回 (docs, ids, source)。解析失败时返回空列表。

    - csv/xlsx: 按 question/human_answers QA 结构解析切片
    - txt/md/docx/pdf: 整体提取文本后按 chunk_size 切片
    """
    ext = Path(filename).suffix.lower()
    log.info("[切片] 开始处理文件: %s (扩展名=%s, 大小=%d, %.2fKB)",
             filename, ext, len(raw), len(raw) / 1024)
    try:
        if ext == ".xlsx":
            text = await asyncio.to_thread(xlsx_to_csv_text, raw)
            rows = load_rows_from_text(text)
            if not rows:
                log.warning("[切片] 文件 %s 未解析到有效知识条目, 跳过", filename)
                return [], [], filename
            docs, ids = build_documents(rows, source=filename)
        elif ext == ".csv":
            rows = load_rows_from_text(_decode_text(raw))
            if not rows:
                log.warning("[切片] 文件 %s 未解析到有效知识条目, 跳过", filename)
                return [], [], filename
            docs, ids = build_documents(rows, source=filename)
        elif ext == ".docx":
            text = await asyncio.to_thread(docx_to_text, raw)
            docs, ids = build_text_documents(filename, text)
        elif ext == ".pdf":
            text = await asyncio.to_thread(pdf_to_text, raw)
            docs, ids = build_text_documents(filename, text)
        elif ext in (".txt", ".md"):
            docs, ids = build_text_documents(filename, _decode_text(raw))
        else:
            log.warning("[切片] 文件 %s 扩展名 %s 不在支持范围, 跳过", filename, ext)
            return [], [], filename
    except Exception as e:  # noqa: BLE001 单个文件解析失败, 跳过该文件继续处理其余
        log.warning("[切片] 文件 %s 解析失败, 已跳过: %s", filename, e)
        return [], [], filename
    log.info("[切片] 文件 %s 切片完成: 生成 %d 个切片", filename, len(docs))
    return docs, ids, filename


@app.post("/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """接收前端上传的普通文档或 zip 压缩包, 解析切片后向量化写入 Chroma。

    - 普通文档(txt/md/csv/xlsx/docx/pdf): 直接解析切片入库
    - zip 压缩包: 递归解压(最多 3 层), 对内部 md/txt/docx/pdf/csv/xlsx 等
      文档逐一提取切片入库; 层级超限的嵌套压缩包直接丢弃
    - 同名来源重复上传时先删旧切片再写入, 实现覆盖更新
    """
    filename = file.filename or "untitled"
    ext = Path(filename).suffix.lower()
    raw = await file.read()
    log.info("[上传] 收到文件: %s (扩展名=%s, 大小=%d, %.2fKB)",
             filename, ext, len(raw), len(raw) / 1024)
    if not raw.strip():
        log.warning("[上传] 文件 %s 内容为空", filename)
        raise HTTPException(400, "文件内容为空")

    is_archive = ext == ".zip" or is_zip(raw)
    size_limit = MAX_ARCHIVE_MB if is_archive else MAX_UPLOAD_MB
    if len(raw) > size_limit * 1024 * 1024:
        log.error("[上传] 文件 %s 大小 %d 超过 %dMB 限制", filename, len(raw), size_limit)
        raise HTTPException(400, f"文件大小超过 {size_limit}MB 限制")

    all_docs, all_ids, sources = [], [], []

    if is_archive:
        # 压缩包: 递归解压后逐一处理内部文档
        log.info("[上传] %s 为压缩包, 开始递归解压", filename)
        try:
            extracted = await asyncio.to_thread(extract_archive, raw)
        except ArchiveError as e:
            log.error("[上传] 压缩包 %s 解压失败: %s", filename, e)
            raise HTTPException(400, f"压缩包解压失败: {e}")
        log.info("[上传] 解压得到可解析文档 %d 个, 开始逐一切片", len(extracted))
        if not extracted:
            raise HTTPException(400, "压缩包内未找到可解析文档(txt/md/csv/xlsx/docx/pdf)")

        skipped = 0
        for i, (inner_name, inner_bytes) in enumerate(extracted, 1):
            log.info("[上传] 处理解压文件 %d/%d: %s", i, len(extracted), inner_name)
            docs, ids, src = await _build_docs_for_file(inner_name, inner_bytes)
            if not docs:
                skipped += 1
                continue
            all_docs.extend(docs)
            all_ids.extend(ids)
            sources.append(src)
        log.info("[上传] 切片汇总: 成功 %d 个文件, 跳过 %d 个, 切片总数 %d",
                 len(sources), skipped, len(all_docs))
        if not all_docs:
            raise HTTPException(400, f"压缩包内 {skipped} 个文档均解析失败, 未生成有效切片")
    else:
        if ext not in ALLOWED_UPLOAD_EXTS:
            log.warning("[上传] 文件 %s 扩展名 %s 不在支持范围", filename, ext)
            raise HTTPException(
                400,
                f"仅支持 {'/'.join(sorted(ALLOWED_UPLOAD_EXTS))} 格式文件",
            )
        docs, ids, src = await _build_docs_for_file(filename, raw)
        if not docs:
            raise HTTPException(400, "未生成任何有效切片")
        all_docs.extend(docs)
        all_ids.extend(ids)
        sources.append(src)

    def _write():
        # 覆盖更新: 删除本次涉及来源的旧切片, 再写入新切片
        log.info("[入库] 开始删除旧切片, 涉及来源 %d 个: %s", len(sources), sources)
        vectorstore._collection.delete(where={"source": {"$in": sources}})
        log.info("[入库] 旧切片删除完成, 开始写入新切片 %d 个", len(all_docs))
        vectorstore.add_documents(all_docs, ids=all_ids)
        count = vectorstore._collection.count()
        log.info("[入库] 写入完成, 当前集合总数 %d", count)
        return count

    total = await asyncio.to_thread(_write)
    log.info("[上传] 全流程完成: 文件=%s, 新增切片=%d, 知识库总数=%d",
             filename, len(all_docs), total)
    return {
        "success": True,
        "filename": filename,
        "added": len(all_docs),
        "total": total,
        "sources": sources,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
