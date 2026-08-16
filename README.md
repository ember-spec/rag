# 离线知识库客服问答系统（RAG）

基于本地 CSV 知识库的 RAG 客服系统：离线脚本构建 Chroma 向量库，FastAPI 后端检索召回 + 大模型流式生成，前端 SSE 打字机渲染。答案严格限定在知识库内容内，知识库没有则回复"暂无相关信息"。

## 架构与数据流

```
docs/file.csv ──> import_csv_to_chroma.py（离线入库）──> chroma_db/（Chroma 本地向量库）
                                                              │
用户提问 ──> index.html ──POST /chat/stream──> main.py（向量检索 top-N 召回）
                                                              │
                     SSE 流式推送 <── 组装 RAG Prompt <── 阿里云大模型（OpenAI 兼容接口）
```

## 项目结构

| 文件 | 说明 |
|---|---|
| `import_csv_to_chroma.py` | 离线入库脚本：csv 解析 → LangChain Document → 切片 → 向量化 → Chroma 持久化 |
| `main.py` | FastAPI 后端：`POST /chat/stream` SSE 流式问答、`POST /upload` 文件上传入库 |
| `index.html` | 客服聊天页面：SSE 打字机渲染、参考来源展示、📎 上传知识文件 |
| `embedding.py` | `AliDashScopeEmbeddings`：通义 text-embedding-v3（OpenAI 兼容接口封装） |
| `docs/file.csv` | 知识库源数据（question / human_answers / chatgpt_answers） |
| `.env` | 配置文件（API Key、模型名、检索参数），**不入库** |

## 快速开始

### 1. 环境准备

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 .env

复制以下内容创建 `.env`，填入你的阿里云 DashScope API Key（[获取地址](https://bailian.console.aliyun.com/)）：

```ini
DASHSCOPE_API_KEY=sk-你的Key

EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3

LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

CSV_PATH=docs/file.csv
CHROMA_DIR=chroma_db
COLLECTION_NAME=rag_pro_kb
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4
SCORE_THRESHOLD=0.35

HOST=0.0.0.0
PORT=8000
```

### 3. 离线入库（构建向量库）

```bash
python import_csv_to_chroma.py             # 增量 upsert（幂等，可重复执行）
python import_csv_to_chroma.py --rebuild   # 清空集合后全量重建
```

### 4. 启动服务

```bash
python main.py
# 打开 http://localhost:8000
```

## 使用说明

- **提问**：输入框输入问题，Enter 发送；回答流式逐字渲染，气泡下方显示命中的知识来源与相关度
- **上传知识**：点 📎 按钮上传 `.txt` / `.md` / `.csv`
  - csv 按 `question,human_answers` 结构解析为 QA 知识
  - txt / md 整篇切片入库
  - 同名文件重复上传为覆盖更新；**上传后即时生效，无需重启**

### API

```bash
# SSE 流式问答
curl -N -X POST http://localhost:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"什么是程序流程图"}'

# 文件上传入库
curl -X POST http://localhost:8000/upload -F "file=@知识库补充.txt"
```

SSE 事件格式：`{"type":"sources",...}`（来源）→ `{"type":"delta","content":"..."}` × N（增量文本）→ `data: [DONE]`（结束）。

## 配置项说明

| 配置 | 默认值 | 说明 |
|---|---|---|
| `TOP_K` | 4 | 召回切片数量 |
| `SCORE_THRESHOLD` | 0.35 | cosine 相关度阈值，低于则判定"暂无相关信息"（不调用大模型，杜绝编造） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 500 / 50 | 切片长度 / 相邻切片重叠字符数 |
| `LLM_MODEL` | qwen-plus | 回答模型，可换 qwen-turbo / qwen-max 等 |

## 注意事项

1. **启动顺序**：先入库、后启动服务；重新执行入库脚本后需**重启服务**（服务持有启动时加载的 Chroma 句柄，感知不到外部进程写入）
2. **网页上传不受此限制**：上传走服务自身进程写入，即时生效
3. `--rebuild` 全量重建会清空全部数据（含网页上传的内容），重建后需重新上传
4. 查看向量库内容：
   ```python
   import chromadb
   col = chromadb.PersistentClient(path='chroma_db').get_collection('rag_pro_kb')
   print(col.count()); col.peek(3)
   ```
