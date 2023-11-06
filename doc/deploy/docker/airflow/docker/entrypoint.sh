#!/bin/bash

export APP_NAME=airflow


if [ ! -f "/opt/airflow/db_init" ]; then
    touch /opt/airflow/db_init
    airflow db init
    airflow users create --username admin --firstname admin --lastname admin --role Admin --email sdataft@163.com --password admin
    echo "airflow init finished"
else
  echo "db_init already existed"
fi

echo "scheduler start"
airflow scheduler -D  >> /opt/airflow/logs/scheduler.log 2>&1 &
echo "scheduler finished"


echo "webserver start"
airflow webserver -p 18080 >> /opt/airflow/logs/webserver.log 2>&1 &
echo "webserver finished"

tail -f /dev/null