"""Sample only an owned GPU worker's process memory, without importing Genesis.

NVIDIA exact-PID compute usage falls back to WDDM Dedicated Usage. WDDM
sums unique all-adapter LUID instances conservatively. Peaks are sampled
observations, never a resident-memory hard-cap proof or device-total proxy.
"""
from __future__ import annotations
import base64
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

HIDDEN=getattr(subprocess,'CREATE_NO_WINDOW',0)
PS=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'


def finite(value):
    if isinstance(value,bool):return None
    try:
        number=float(value)
        return number if math.isfinite(number) and number>=0 else None
    except (TypeError,ValueError):return None


def units(value):
    return None if value is None else dict(bytes=value,GB=value/1e9,GiB=value/1024**3)


def command(argv,timeout):
    started=time.perf_counter()
    try:
        r=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8-sig',
            errors='replace',timeout=timeout,creationflags=HIDDEN,check=False)
        return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,elapsed_s=time.perf_counter()-started)
    except Exception as exc:
        return dict(error=type(exc).__name__+': '+str(exc),elapsed_s=time.perf_counter()-started)


def parse_nvidia(text,pid):
    unique={}
    for row in csv.reader(io.StringIO(text)):
        if len(row)!=3 or row[0].strip()!=str(pid):continue
        uuid=row[2].strip()
        if not uuid:continue
        mib=finite(row[1].strip())
        value=dict(pid=pid,gpu_uuid=uuid,raw_memory=row[1].strip(),
            bytes=None if mib is None else mib*1024**2)
        if uuid in unique and unique[uuid]['bytes']!=value['bytes']:
            raise ValueError('Conflicting duplicate NVIDIA PID/GPU row')
        unique[uuid]=value
    return list(unique.values())


def parse_windows(value,pid):
    if type(value) is not dict or type(value.get('pid')) is not int or value['pid']!=pid:return []
    rows=value.get('samples',[])
    if isinstance(rows,dict):rows=[rows]
    unique={}
    for row in rows:
        if type(row) is not dict:continue
        name=str(row.get('instance','')).lower()
        amount=finite(row.get('bytes'))
        if (name.startswith('pid_'+str(pid)+'_') and type(row.get('status')) is int
                and row['status']==0 and amount is not None and amount==int(amount)):
            if name in unique and unique[name]['bytes']!=amount:
                raise ValueError('Conflicting duplicate Windows counter instance')
            unique[name]=dict(instance=name,status=0,bytes=amount)
    return list(unique.values())


def creation_identity(process):
    if os.name!='nt':raise RuntimeError('Exact Windows Popen handle identity is required')
    created,exited,kernel,user=(wintypes.FILETIME() for _ in range(4))
    fn=ctypes.windll.kernel32.GetProcessTimes
    fn.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    fn.restype=wintypes.BOOL
    if not fn(wintypes.HANDLE(int(process._handle)),ctypes.byref(created),ctypes.byref(exited),
              ctypes.byref(kernel),ctypes.byref(user)):raise ctypes.WinError()
    return dict(pid=process.pid,windows_creation_filetime_100ns=(created.dwHighDateTime<<32)+created.dwLowDateTime)


def windows_query(pid):
    script=("$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        f"$taskPid={int(pid)}; "
        "$taskRows=@((Get-Counter -Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples | "
        "Where-Object { $_.InstanceName -match ('^pid_'+$taskPid+'_') -and $_.Status -eq 0 } | "
        "ForEach-Object { [pscustomobject]@{ instance=$_.InstanceName; status=$_.Status; bytes=$_.CookedValue } }); "
        "[pscustomobject]@{ pid=$taskPid; samples=$taskRows; utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json -Compress -Depth 4")
    encoded=base64.b64encode(script.encode('utf-16le')).decode('ascii')
    return command([str(PS),'-NoProfile','-NonInteractive','-EncodedCommand',encoded],5.)


def kill_owned_tree(process,identity):
    """Terminate only the still-live owned worker and its descendants."""
    if process.poll() is not None:return None
    if creation_identity(process)!=identity:raise RuntimeError('Owned worker identity changed before tree termination')
    if os.name=='nt':
        result=command([str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/taskkill.exe'),
            '/PID',str(process.pid),'/T','/F'],5.)
    else:result=dict(scope='Non-Windows tree termination is not supported')
    if process.poll() is None:process.kill()
    process.wait(timeout=3.)
    return result


def summarize(samples,*,limit_gb,worker_exited,monitor_failure=None,elapsed_s=None):
    valid=[s for s in samples if s.get('process_measurement_valid') is True]
    nonzero=[s for s in valid if s['process_bytes']>0]
    peak=max((s['process_bytes'] for s in valid),default=None)
    times=[s['elapsed_s'] for s in valid]
    intervals=[b-a for a,b in zip(times,times[1:])]
    enough=len(nonzero)>=2 and worker_exited and monitor_failure is None
    peak_sample=max(valid,key=lambda s:s['process_bytes']) if valid else None
    return dict(status='SAMPLED_PROCESS_MEMORY_RECORDED' if enough else 'INSUFFICIENT_VALID_PROCESS_SAMPLES',
        sample_count=len(samples),valid_sample_count=len(valid),nonzero_valid_sample_count=len(nonzero),
        unknown_or_invalid_sample_count=len(samples)-len(valid),sampled_peak=units(peak),
        requested_limit=units(limit_gb*1e9),observed_sampled_peak_within_limit=bool(enough and peak<=limit_gb*1e9),
        first_valid_sample_s=None if not times else times[0],last_valid_sample_s=None if not times else times[-1],
        peak_sample_elapsed_s=None if peak_sample is None else peak_sample['elapsed_s'],
        peak_sample_utc=None if peak_sample is None else peak_sample.get('utc'),
        peak_measurement_source=None if peak_sample is None else peak_sample.get('process_measurement_source'),
        maximum_valid_sample_gap_s=max(intervals,default=None),
        startup_to_first_valid_gap_s=None if not times else times[0],
        last_valid_to_exit_gap_s=None if not times or elapsed_s is None else max(0.,elapsed_s-times[-1]),
        worker_exited=worker_exited,monitor_failure=monitor_failure,
        peak_is_sampled_not_true_peak=True,hard_cap_proven=False,
        scope='Exact owned PID+creation identity from startup through worker exit. WDDM is a conservative sum of unique all-adapter Dedicated Usage instances. Unknown gaps remain unknown. Device totals and transfer bytes are never process VRAM.')


class ProcessMemorySampler:
    def __init__(self,process,path,*,started,interval_s=.5,limit_gb=5.,identity_reader=creation_identity,
                 query_command=command,query_windows=windows_query):
        if not .1<=interval_s<=3. or not 0<limit_gb<=5.:raise ValueError('Use sampling0.1..3s and a positive limit at most5 decimal GB')
        self.process=process;self.path=Path(path);self.started=started
        self.interval_s=interval_s;self.limit_gb=limit_gb
        self.identity_reader=identity_reader;self.query_command=query_command;self.query_windows=query_windows
        self.identity=identity_reader(process)
        self.samples=[];self.failure=None;self.stop=threading.Event();self.thread=None
        self.finished=False;self.exit_elapsed_s=None
        self.limit_exceeded=threading.Event();self.first_limit_breach=None

    def start(self):
        if self.thread is not None:raise RuntimeError('Memory sampler is one-use')
        self.thread=threading.Thread(target=self._safe_collect,daemon=True)
        self.thread.start()
        return self.identity

    def _safe_collect(self):
        try:self._collect()
        except BaseException as exc:
            self.failure=dict(type=type(exc).__name__,message=str(exc))
            self.stop.set()

    def _collect(self):
        last_windows=-math.inf
        with self.path.open('x',encoding='utf-8') as stream:
            while not self.stop.is_set() and self.process.poll() is None:
                tick=time.perf_counter()
                sample=dict(elapsed_s=tick-self.started,utc=datetime.now(timezone.utc).isoformat(),
                    owned_worker=self.identity,worker_alive_before=True,process_measurement_valid=False,
                    process_bytes=None,process_measurement_source=None)
                if self.identity_reader(self.process)!=self.identity:
                    raise RuntimeError('Owned worker creation identity changed')
                query=self.query_command(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid',
                    '--format=csv,noheader,nounits'],3.)
                sample['nvidia_process_query']=query
                rows=parse_nvidia(query.get('stdout',''),self.process.pid)
                sample['nvidia_matching_process_rows']=rows
                if query.get('exit_code')==0 and rows and all(r['bytes'] is not None for r in rows):
                    sample.update(process_measurement_valid=True,process_bytes=sum(r['bytes'] for r in rows),
                        process_measurement_source='NVIDIA exact-PID compute memory across unique reported GPU UUIDs')
                if not sample['process_measurement_valid'] and tick-last_windows>=3.:
                    last_windows=tick;fallback=self.query_windows(self.process.pid)
                    sample['windows_process_query']=fallback
                    try:
                        value=json.loads(fallback.get('stdout','')) if fallback.get('exit_code')==0 else {}
                        found=parse_windows(value,self.process.pid);sample['windows_counter_instances']=found
                        if found:
                            sample.update(process_measurement_valid=True,process_bytes=sum(r['bytes'] for r in found),
                                process_measurement_source='Windows exact-PID Dedicated Usage: conservative unique all-adapter LUID sum')
                    except Exception as exc:sample['windows_counter_error']=type(exc).__name__+': '+str(exc)
                sample['worker_alive_after']=self.process.poll() is None
                sample['owned_identity_unchanged']=self.identity_reader(self.process)==self.identity
                if not sample['worker_alive_after'] or not sample['owned_identity_unchanged']:
                    sample['process_measurement_valid']=False
                if sample['process_measurement_valid'] and finite(sample['process_bytes']) is None:
                    sample['process_measurement_valid']=False
                sample['sample_duration_s']=time.perf_counter()-tick
                self.samples.append(sample)
                stream.write(json.dumps(sample,allow_nan=False)+'\n');stream.flush()
                if (sample['process_measurement_valid'] and sample['process_bytes']>self.limit_gb*1e9
                        and self.first_limit_breach is None):
                    self.first_limit_breach=dict(elapsed_s=sample['elapsed_s'],
                        process_bytes=sample['process_bytes'],limit_bytes=self.limit_gb*1e9,
                        source=sample['process_measurement_source'],owned_worker=self.identity)
                    self.limit_exceeded.set()
                self.stop.wait(max(0.,self.interval_s-sample['sample_duration_s']))
            stream.flush();os.fsync(stream.fileno())

    def finish(self):
        if self.finished:raise RuntimeError('Memory sampler cannot finish twice')
        self.exit_elapsed_s=time.perf_counter()-self.started
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=9.)
        if self.thread is not None and self.thread.is_alive():
            self.failure=dict(type='MonitorStopTimeout',message='Memory monitor did not stop within9s')
        self.finished=True
        receipt=summarize(self.samples,limit_gb=self.limit_gb,worker_exited=self.process.poll() is not None,
            monitor_failure=self.failure,elapsed_s=self.exit_elapsed_s)
        receipt.update(schema_version='owned-process-vram-samples-v1',owned_worker=self.identity,
            first_observed_limit_breach=self.first_limit_breach,
            requested_interval_s=self.interval_s,windows_fallback_minimum_spacing_s=3.,
            raw_samples_path=str(self.path),sampler_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            raw_samples_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.is_file() and not (self.thread and self.thread.is_alive()) else None)
        return receipt


def wait_owned_worker(process,memory,identity,*,deadline):
    """Stop only the owned process tree on deadline or a valid sampled breach."""
    termination=None;timed_out=False;memory_limit_stop=False
    while process.poll() is None:
        if memory.limit_exceeded.is_set():
            memory_limit_stop=True;termination=kill_owned_tree(process,identity);break
        remaining=deadline-time.perf_counter()
        if remaining<=0:
            timed_out=True;termination=kill_owned_tree(process,identity);break
        try:process.wait(timeout=min(.1,remaining))
        except subprocess.TimeoutExpired:pass
    return dict(exit_code=process.poll(),timed_out=timed_out,
        memory_limit_stop=memory_limit_stop,termination=termination)

