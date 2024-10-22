"""
Create a cluster of GPU nodes to perform parallel prediction of tiles
"""
import sys
import socket
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import gc
import os
from deepforest import main
from src import model
from dask.distributed import wait
from dask import delayed

def collect():
    gc.collect()

def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()
    print("To tunnel into dask dashboard:")
    print("For GPU dashboard: ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" %
          (host))
    print("For CPU dashboard: ssh -N -L 8781:%s:8781 -l b.weinstein hpg2.rc.ufl.edu" %
          (host))

    #flush system
    sys.stdout.flush()


def start(cpus=0, gpus=0, mem_size="50GB"):
    #################
    # Setup dask cluster
    #################

    if cpus > 0:
        #job args
        extra_args = [
            "--error=/home/b.weinstein/logs/dask-worker-%j.err", "--account=ewhite",
            "--output=/home/b.weinstein/logs/dask-worker-%j.out"
        ]

        cluster = SLURMCluster(processes=1,
                               queue='hpg2-compute',
                               cores=1,
                               memory=mem_size,
                               walltime='24:00:00',
                               job_extra=extra_args,
                               extra=['--resources cpu=1'],
                               scheduler_options={"dashboard_address": ":8781"},
                               local_directory="/orange/idtrees-collab/tmp/",
                               death_timeout=300)

        print(cluster.job_script())
        cluster.scale(cpus)
        cluster.wait_for_workers(cpus)

    if gpus:
        #job args
        extra_args = [
            "--error=/home/b.weinstein/logs/dask-worker-%j.err", "--account=ewhite",
            "--output=/home/b.weinstein/logs/dask-worker-%j.out", "--partition=gpu",
            "--gpus=1"
        ]

        cluster = SLURMCluster(processes=1,
                               cores=2,
                               memory=mem_size,
                               walltime='24:00:00',
                               job_extra=extra_args,
                               extra=['--resources gpu=1'],
                               nanny=False,
                               scheduler_options={"dashboard_address": ":8787"},
                               local_directory="/orange/idtrees-collab/tmp/",
                               death_timeout=10000)
        print(cluster.job_script())
        cluster.scale(gpus)
        # Wait for atleast half the workers
        cluster.wait_for_workers(2)

    dask_client = Client(cluster)

    #Start dask
    dask_client.run_on_scheduler(start_tunnel)

    return dask_client

def create_model(model_checkpoint, checkpoint_dir, annotations):
    if model_checkpoint:
        m = model.load(model_checkpoint)
    elif os.path.exists(checkpoint_dir):
        m = model.get_latest_checkpoint(checkpoint_dir, annotations)
    else:
        m = main.deepforest()
        m.use_bird_release()
    return m
