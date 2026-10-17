# 项目 B 独立镜像；使用 Docker 官方 Python 镜像在 ECR Public 的镜像地址。
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.lock.txt /app/
# 依赖层只由锁定文件决定，修改业务代码、README 或项目元数据时可复用构建缓存。
RUN sed '/^-e \./d' requirements.lock.txt > /tmp/requirements-third-party.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-third-party.txt
COPY pyproject.toml README.md LICENSE /app/
COPY corecoder /app/corecoder
COPY skillflow /app/skillflow
COPY data /app/data
# 容器固定从 /app 启动，Python 可直接导入此处的项目 B 包；避免重复安装项目时再次联网下载构建后端。
RUN groupadd --gid 10001 skillflow \
    && useradd --uid 10001 --gid 10001 --home-dir /app --no-create-home skillflow \
    && mkdir -p /app/.skillflow \
    && chown -R skillflow:skillflow /app

# 默认命令仅用于隔离端口的 demo 冒烟；生产 HTTPS 参数由 compose.local.yaml 明确覆盖。
USER 10001:10001
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "skillflow.api:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
