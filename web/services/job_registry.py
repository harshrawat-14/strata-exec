"""
In-memory registry for tracking active asyncio Tasks and running subprocesses.
Allows clean cancellation and process termination of jobs.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Set

# Global maps for the current process
_active_tasks: Dict[str, asyncio.Task] = {}
_active_subprocesses: Dict[str, Set[asyncio.subprocess.Process]] = {}
_registry_lock = asyncio.Lock()


async def register_task(job_id: str, task: asyncio.Task):
    """Register the main task running a job."""
    async with _registry_lock:
        _active_tasks[job_id] = task


async def unregister_task(job_id: str):
    """Clean up task and subprocess records for a job."""
    async with _registry_lock:
        _active_tasks.pop(job_id, None)
        _active_subprocesses.pop(job_id, None)


async def register_subprocess(job_id: str, proc: asyncio.subprocess.Process):
    """Register a subprocess spawned by a job."""
    async with _registry_lock:
        if job_id not in _active_subprocesses:
            _active_subprocesses[job_id] = set()
        _active_subprocesses[job_id].add(proc)


async def unregister_subprocess(job_id: str, proc: asyncio.subprocess.Process):
    """Unregister a completed subprocess."""
    async with _registry_lock:
        if job_id in _active_subprocesses:
            _active_subprocesses[job_id].discard(proc)
            if not _active_subprocesses[job_id]:
                _active_subprocesses.pop(job_id, None)


async def abort_job_local(job_id: str):
    """
    Abort a job locally:
    1. Terminate all active subprocesses associated with the job.
    2. Cancel the main asyncio Task.
    """
    async with _registry_lock:
        # 1. Terminate subprocesses first
        procs = _active_subprocesses.get(job_id, set())
        for proc in list(procs):
            try:
                proc.terminate()
            except Exception:
                pass
        
        # 2. Cancel the task
        task = _active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
