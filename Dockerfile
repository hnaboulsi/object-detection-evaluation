FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    MUJOCO_GL=osmesa \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/cache

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libegl1 \
        libgl1 \
        libglfw3 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements-render.txt requirements.txt ./
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "render_experiment.py", "--help"]

FROM runtime AS test
RUN python -m pip install --no-cache-dir -r requirements-dev.txt
CMD ["python", "-m", "pytest", "-q"]
