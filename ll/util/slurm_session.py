def get_current_slurm_session_info():
    try:
        from submitit import JobEnvironment

        job = JobEnvironment()
        if not job.activated():
            return {}

        return {
            "hostname": job.hostname,
            "hostnames": job.hostnames,
            "job_id": job.job_id,
            "raw_job_id": job.raw_job_id,
            "array_job_id": job.array_job_id,
            "array_task_id": job.array_task_id,
            "num_tasks": job.num_tasks,
            "num_nodes": job.num_nodes,
            "node": job.node,
            "global_rank": job.global_rank,
            "local_rank": job.local_rank,
        }
    except (ImportError, RuntimeError):
        return {}
