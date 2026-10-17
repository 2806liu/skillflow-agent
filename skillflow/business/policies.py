"""项目 B 自有政策库：在本项目内索引 Markdown，按商品政策范围进行词法检索。"""
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from skillflow.contracts import Evidence, ToolResult


def tokenize(text: str) -> list[str]:
    """英文标识符按词分割；中文使用二元字组，减少对分词模型和外部服务的依赖。"""
    tokens = []
    for term in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            tokens.extend(term[index:index + 2] for index in range(max(1, len(term) - 1)))
        else:
            tokens.append(term)
    return tokens


class LocalPolicyRepository:
    """每个实例只持有项目 B 的政策索引；没有网络客户端或跨项目数据源。"""

    def __init__(self, policy_dir: Path):
        self.policy_dir = policy_dir
        self.documents = {}
        self.chunks = []
        self.reload()

    def reload(self):
        """重建本地索引；文档和片段 ID 由内容生成，方便引用回溯与重启复现。"""
        documents, chunks = {}, []
        for path in sorted(self.policy_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            keys = set(re.findall(r"(?<!\w)policy_key:\s*([a-zA-Z0-9_]+)(?!\w)", text))
            # 一份政策只能属于一个明确的商品政策范围，避免引用到其他商品的规则。
            if len(keys) != 1:
                continue
            policy_key = keys.pop()
            document_id = "policy-" + hashlib.sha256((path.name + text).encode()).hexdigest()[:24]
            documents[document_id] = {"path": path, "text": text, "policy_key": policy_key}
            # 以段落切块，保留原始换行；长段落进一步拆分，确保片段尺寸有上限。
            blocks, current = [], ""
            for paragraph in text.split("\n\n"):
                if current and len(current) + len(paragraph) + 2 > 1200:
                    blocks.append(current)
                    current = ""
                current += ("\n\n" if current else "") + paragraph
                while len(current) > 1600:
                    blocks.append(current[:1600])
                    current = current[1600:]
            if current:
                blocks.append(current)
            for position, snippet in enumerate(blocks):
                tokens = tokenize(snippet)
                chunks.append({"document_id": document_id, "policy_key": policy_key,
                               "chunk_id": f"{document_id}-{position}", "filename": path.name,
                               "snippet": snippet, "terms": Counter(tokens), "length": len(tokens)})
        # 完成全部读取后再替换索引，读取失败时不发布半成品。
        self.documents, self.chunks = documents, chunks

    def registered_policies(self):
        """返回本项目已索引的政策及文档 ID，不读取以前的跨服务映射。"""
        return {doc["policy_key"]: document_id for document_id, doc in self.documents.items()}

    async def retrieve(self, policy_key: str, query: str):
        """先按商品范围筛选，再用 BM25 对本地片段排序；不命中时不生成证据。"""
        candidates = [chunk for chunk in self.chunks if chunk["policy_key"] == policy_key]
        if not candidates:
            return ToolResult(status="empty", error_code="POLICY_NOT_REGISTERED",
                              message="该商品没有本地政策文档，无法判断退换条件。")
        query_terms = set(tokenize(query))
        average_length = sum(chunk["length"] for chunk in candidates) / len(candidates) or 1
        ranked = []
        for chunk in candidates:
            score = 0.0
            for term in query_terms:
                frequency = chunk["terms"].get(term, 0)
                if not frequency:
                    continue
                document_frequency = sum(term in other["terms"] for other in candidates)
                idf = math.log(1 + (len(candidates) - document_frequency + 0.5) / (document_frequency + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * chunk["length"] / average_length))
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        evidence = [Evidence(citation_number=index, document_id=chunk["document_id"],
                             chunk_id=chunk["chunk_id"], filename=chunk["filename"], page=None,
                             snippet=chunk["snippet"], score=round(score, 6)).model_dump()
                    for index, (score, chunk) in enumerate(ranked[:4], 1)]
        return ToolResult(status="success" if evidence else "empty",
                          data={"evidence": evidence, "retrieval_trace": {"strategy": "local_bm25_v1", "candidate_count": len(candidates), "source": "project_b_local"}},
                          message="已检索项目 B 本地政策库。" if evidence else "本地政策库未检索到可引用的片段。")

    def source(self, document_id: str):
        """只提供当前索引中的原文件；用户不能通过参数指定任意文件路径。"""
        document = self.documents.get(document_id)
        return document["path"] if document else None
