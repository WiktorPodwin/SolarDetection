FROM postgres:14

USER root

# Install dependencies
RUN apt-get update && apt-get install -y \
  wget \
  gnupg \
  unzip \
  curl \
  xvfb \
  && rm -rf /var/lib/apt/lists/*
  
USER postgres

EXPOSE 5432
