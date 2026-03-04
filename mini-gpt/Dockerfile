FROM ubuntu:22.04 

RUN apt-get update -y \
    && apt-get install -y python3 python3-pip \
    && apt-get -qq purge && apt-get -qq clean && rm -rf /var/lib/apt/lists/*


#RUN pip3 install --no-cache-dir -qq transformers torch accelerate datasets
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers accelerate datasets


WORKDIR /root
COPY ./tiny_gpt2_cpu_1k.py ./tiny_gpt2_cpu_1k.py
