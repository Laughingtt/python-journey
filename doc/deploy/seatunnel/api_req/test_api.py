import time

import requests
import json
import os



os.environ['prod_mysql_user']="1"
os.environ['prod_mysql_password']="2"
os.environ['stg_mysql_user']="3"
os.environ['stg_mysql_password']="4"


def get_job_running(job_id):
    url = "http://1127.0.0.1:5801/hazelcast/rest/maps/job-info/{job_id}".format(job_id=job_id)
    payload = {}
    headers = {}

    is_running = True
    t0 = time.time()
    timeout = 100
    while is_running:
        response = requests.request("GET", url, headers=headers, data=payload)

        print(response.json())
        jobStatus = response.json().get("jobStatus", None)
        if jobStatus == "FINISHED":
            is_running = False
        else:
            time.sleep(1)
            if time.time() - t0 >= timeout:
                is_running = False
                raise ValueError("timeout or running")
    print("task job {} finished".format(job_id))


def submit_job(sea_json):
    url = "http://127.0.0.1:5801/hazelcast/rest/maps/submit-job"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = json.dumps(sea_json)

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.json())
    return response


def replace_vars_in_json(template: str):
    replaced_template = template
    for key, value in os.environ.items():
        replaced_template = replaced_template.replace(f'${{{key}}}', value)
    return json.loads(replaced_template)


def batch_runner_prod2stg(**kwargs):
    json_path = "/data/airflow/dataflow/src/dags/utils/seatunnel/jobs/batch_runner_prod2stg_ps.json"
    with open(json_path, 'r', encoding='utf-8') as file:
        sea_rd = file.read()

    sea_json=replace_vars_in_json(sea_rd)
    print(sea_json)

    response = submit_job(sea_json)
    jobid = response.json().get("jobId", None)

    if jobid is None:
        return None
    get_job_running(jobid)

    return None


batch_runner_prod2stg()
