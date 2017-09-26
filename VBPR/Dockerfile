FROM ruby:2.2.5
MAINTAINER Alex Egg <eggie5@gmail.com>

RUN apt-get update && apt-get install -y \
    libpq-dev \
    build-essential \
    nodejs \
    qt5-default \
    wget\
    python2.7-dev \
    vim \
    libarmadillo-dev
    
ENV APP_HOME .
WORKDIR $APP_HOME

COPY . $APP_HOME/


#docker run -v ~/Development/DSE/capstone/UpsDowns:/mnt/mac  -ti updowns /bin/bash