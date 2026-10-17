"""只启动项目 B；政策、数据和配置均来自本项目。"""
import os
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def schema(url):
    """通过接口定义判断 B 是否已就绪；禁用代理以直接访问本机端口。"""
    try:
        # 只检查当前项目的服务，不探测其他项目。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/openapi.json", timeout=3) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def health(url):
    """读取项目 B 健康信息，用于拒绝复用旧版本服务，避免浏览器继续命中旧路由。"""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/health", timeout=3) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def main():
    """优先复用已经启动的独立 B 服务，否则启动本项目虚拟环境的 Uvicorn。"""
    children = []
    try:
        configs = [
            (ROOT, "skillflow.api:app", 8001, {})]
        for directory, module, port, overrides in configs:
            url = f"http://127.0.0.1:{port}"
            existing = schema(url)
            if existing:
                paths = existing["paths"]
                # 旧版跨项目服务不可复用，需要先由其启动者关闭并启动新版。
                # 旧 Python 进程会读取新网页文件，但不会自动加载新增的 API 路由。
                # 检查当前工单接口，避免新网页连接到旧后端。
                required = {"/policies/{document_id}/source", "/tickets/{ticket_id}/cancel"}
                if not required <= set(paths):
                    raise RuntimeError(
                        f"Port {port} is running an older or incompatible SkillFlow service; "
                        "stop that process and start the demo again")
                current_health = health(url)
                # 仅有旧接口并不能证明已加载结构化对话路由；启动脚本必须阻止
                # 复用旧进程，否则修改 planner/runtime 后网页仍会显示旧答案。
                # B4 只复用已加载 SQLite 检查点的进程，防止验收落到仍在运行的旧服务。
                if (not current_health or current_health.get("dialogue_version") != "structured_dialogue_v1"
                        or current_health.get("session_mode") != "sqlite_langgraph_checkpointer"):
                    raise RuntimeError(
                        f"Port {port} is running an old SkillFlow dialogue service; "
                        "stop that process and start the demo again")
                print(f"Reusing {url}", flush=True)
                continue
            python = directory / ".venv/Scripts/python.exe"
            if not python.exists():
                raise RuntimeError(f"Missing Python environment: {python}")
            child = subprocess.Popen([str(python), "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port)],
                                     cwd=directory, env={**os.environ, **overrides})
            children.append(child)
            for _ in range(40):
                if child.poll() is not None:
                    raise RuntimeError(f"Service on {port} failed to start")
                if schema(url):
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(f"Service on {port} did not become ready")
        print("聊天页面：http://127.0.0.1:8001/；政策已在本项目启动时完成索引。", flush=True)
        if not children:
            return
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # 仅清理本脚本创建的子进程，绝不关闭其他项目或用户已有的服务。
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()


if __name__ == "__main__":
    main()
