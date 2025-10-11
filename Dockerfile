# 1. Base Image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 2. Munkakönyvtár
WORKDIR /app

# 3. Rendszerfüggőségek
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Python csomagok
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Forráskód
COPY ./src .

# 6. Script futtathatóvá tétele
RUN chmod +x run.sh

# 7. Default parancs
# 8. Jupyter beállítás: jelszó és token kikapcsolása
RUN jupyter server --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.allow_origin = '*'" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_server_config.py



CMD ["bash", "run.sh"]
