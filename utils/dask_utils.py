# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2022
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import cdsw

# This default may not work in some envs - change to whatever is appropriate 
DASHBOARD_PORT = os.environ['CDSW_READONLY_PORT']


def run_scheduler(dashboard_port, num_workers=1, cpu=1, memory=2):
    """
    Run a Dask Scheduler process in a CDSW worker.
    """
    scheduler_code = f"""!dask-scheduler \
        --host 0.0.0.0 \
        --dashboard-address 127.0.0.1:{dashboard_port}""" 
    dask_scheduler = cdsw.launch_workers(
        n=num_workers,
        cpu=cpu,
        memory=memory,
        code=scheduler_code,
    )
    scheduler_details = cdsw.await_workers(
      dask_scheduler, 
      wait_for_completion=False, 
      timeout_seconds=90
    )
    if scheduler_details['failures']:
        raise RuntimeError("dask-scheduler worker node failed to launch.")
        print(scheduler_details['failures'][0])
    return dask_scheduler

  
def get_scheduler_url(dask_scheduler):
    """
    Given a Dask Scheduler, identify its TCP url so Dask Workers can 
    communicate with it. 
    """
    scheduler_workers = cdsw.list_workers()
    scheduler_id = dask_scheduler[0]["id"]
    scheduler_ip = [
        worker["ip_address"] for worker in scheduler_workers if worker["id"] == scheduler_id
    ][0]

    return f"tcp://{scheduler_ip}:8786"


def get_dashboard_url(dask_scheduler):
    """ Given a Dask Scheduler, return the Dask Dashboard url"""
    return("//".join(dask_scheduler[0]["app_url"].split("//")) + "status")
  

def run_dask_workers(scheduler_url, num_workers, cpu, memory, nvidia_gpu=0):
    """
    Launch num_workers CDSW workers as Dask Worker nodes.
    Assumes that the Dask Scheduler is running and available via 
    the scheduler_url.
    """
    workers = cdsw.launch_workers(
            n=num_workers, 
            cpu=cpu, 
            memory=memory, 
            nvidia_gpu=nvidia_gpu, 
            code=f"!dask-worker {scheduler_url}"
          )

    worker_details = cdsw.await_workers(workers, wait_for_completion=False)
    if worker_details['failures']:
        raise RuntimeError("dask worker nodes failed to launch.")
        print(worker_details['failures'])
    return workers


def run_dask_cluster(num_workers, cpu, memory, nvidia_gpu=0, dashboard_port=DASHBOARD_PORT):
    """
    Runs a Dask Scheduler and the requested number of Dask Workers
    as CDSW workers via the Workers API. 
    """
    dask_scheduler = run_scheduler(dashboard_port=dashboard_port)
    scheduler_url = get_scheduler_url(dask_scheduler)
    dashboard_url = get_dashboard_url(dask_scheduler)
    dask_workers = run_dask_workers(
        scheduler_url=scheduler_url,
        num_workers=num_workers, 
        cpu=cpu, 
        memory=memory, 
        nvidia_gpu=nvidia_gpu, 
    )
    return {
        "scheduler": dask_scheduler,
        "workers": dask_workers,
        "scheduler_address": scheduler_url,
        "dashboard_address": dashboard_url
        }