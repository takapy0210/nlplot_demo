FROM python:3.10

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Timezone jst
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# Locale Japanese
ENV LC_ALL=ja_JP.UTF-8

# pip
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U pip && \
    python3 -m pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt && \
    rm -rf /root/.cache

# Set up the program in the image
COPY app /opt/program
WORKDIR /opt/program
