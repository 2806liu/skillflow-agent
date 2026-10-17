"""只访问项目 B 的 HTTP 验收：本地政策、订单、引用和执行记录。"""
import argparse
import json
from pathlib import Path
import uuid

import httpx


def main():
    """运行固定正常与异常案例；所有请求均发往同一项目 B 地址。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    results = []
    with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
        health = client.get("/health").json()
        assert health["policy_source"] == "project_b_local", health
        assert "rag_url" not in health, health
        setup = client.post("/demo/setup-policy")
        setup.raise_for_status()
        # 验证正常、有条件限制、无政策、无订单和越权订单的不同终止路径。
        for order, expected in [("A123", "completed"), ("A124", "completed"), ("A125", "insufficient_evidence"), ("A126", "completed"), ("A999", "order_not_found"), ("A900", "order_not_found")]:
            response = client.post("/chat", json={"session_id": str(uuid.uuid4()), "message": f"查询订单 {order}，结合店铺政策，调查这件商品能否退换。"})
            response.raise_for_status()
            data = response.json()
            assert data["status"] == expected, data
            if expected == "completed":
                assert data["citations"] and "[1]" in data["answer"], data
                assert data["result"]["retrieve_policy"]["retrieval_trace"]["source"] == "project_b_local"
                document_id = data["citations"][0]["document_id"]
                assert client.get(f"/policies/{document_id}/source").status_code == 200
            if order == "A124":
                assert "缺货" in data["answer"]
            if order == "A126":
                assert not data["result"]["assess_after_sales"]["return_eligible"]
            assert client.get("/runs/" + data["run_id"]).json() == data
            results.append(data)
        # 多轮追问必须沿用原调查目标，不能将补订单号错误降级成普通查询。
        session = str(uuid.uuid4())
        assert client.post("/chat", json={"session_id": session, "message": "这件商品能否退换？"}).json()["status"] == "clarify"
        continued = client.post("/chat", json={"session_id": session, "message": "A123"}).json()
        # 多轮追问先确认商品；确认后必须恢复原退换任务，不能只降级成订单查询。
        assert continued["status"] == "clarify" and continued["dialogue"]["order_id"] == "A123", continued
        confirmed = client.post("/chat", json={"session_id": session, "message": "对的"}).json()
        assert confirmed["status"] == "completed" and confirmed["plan"]["skill"] == "return_exchange", confirmed
        results.append(confirmed)
        assert client.get("/policies/missing-document/source").status_code == 404
    # 与旧跨项目报告区分，避免把此前集成测试误当作当前独立架构验收。
    output = Path(__file__).resolve().parent.parent / ".skillflow" / "standalone-integration-results.json"
    output.write_text(json.dumps({"health": health, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: 7 standalone HTTP chat cases, product confirmation and local citation sources. Saved: {output}")


if __name__ == "__main__":
    main()
