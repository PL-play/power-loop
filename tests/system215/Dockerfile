FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iptables iproute2 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir asyncpg aiomysql aiokafka redis aiohttp httpx pyyaml openai
WORKDIR /sdk
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/sdk:/agent
CMD ["python", "-m", "tests.system215.node_runner"]
