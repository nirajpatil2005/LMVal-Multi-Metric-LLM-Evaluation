FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install packages with YOUR versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    langchain==0.3.27 \
    langchain-core==0.3.74 \
    langchain-community==0.3.27 \
    langchain-openai==0.3.28 \
    langgraph==0.6.3 \
    langchain-groq==0.3.6 \
    groq==0.30.0 \
    diskcache==5.6.3\
    streamlit==1.37.1 \
    pandas==2.1.3 \
    numpy==1.26.4 \
    pydantic==2.8.2 \
    plotly==5.24.1 \
    nest-asyncio==1.6.0 \
    python-dotenv==0.21.0 \
    tqdm==4.66.5 \
    openai==1.97.0

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]