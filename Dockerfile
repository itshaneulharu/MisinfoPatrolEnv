FROM python:3.10-slim

# Create non-root user (HF compatible)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Set permissions
RUN chown -R appuser:appuser /app
USER appuser

# Hugging Face required port
EXPOSE 7860

# (Optional but safer) remove healthcheck — HF already does this
# HEALTHCHECK removed to avoid startup issues

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]