FROM apache/airflow:2.7.1-python3.10
USER root

RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
  apt update -y && \
  apt install -y --no-install-recommends vim && \
  apt autoremove -yqq --purge && \
  apt install -y procps && \
  apt clean && \
  rm -rf /var/lib/apt/lists/* && \
  ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone

USER airflow

WORKDIR /opt/airflow

#COPY docker/opt/airflow.cfg /opt/airflow
COPY docker/entrypoint.sh /opt/airflow/entrypoint.sh

ENTRYPOINT bash entrypoint.sh

#EXPOSE 18080
