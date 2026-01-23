# 多階段構建：構建階段
FROM python:3.11.11-slim AS builder

# 設定工作目錄
WORKDIR /app

# 安裝構建時需要的系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements 文件
COPY requirements.txt ./

# 安裝 Python 依賴到臨時目錄 (/root/.local)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# 運行階段：只包含運行時需要的文件
FROM python:3.11.11-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=/root/.local/bin:$PATH

# 只安裝運行時需要的系統依賴（移除 build-essential, git, vim 等開發工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 從構建階段複製已安裝的 Python 包
COPY --from=builder /root/.local /root/.local

# 複製應用程式代碼
COPY . .

# 創建必要的目錄
RUN mkdir -p /app/tmp /app/data/documents/file_content /app/data/documents/processed_output/merged_files /app/models

# 設定權限
RUN chmod +x /app/scripts/*.py 2>/dev/null || true

# 暴露端口
EXPOSE 18299

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:18299/api/v1/info || exit 1

# 啟動命令
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "18299"]
