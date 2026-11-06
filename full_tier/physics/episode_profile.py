"""Opt-in short-episode wall profiling. Never a mission acceptance result."""
from collections import defaultdict
from time import perf_counter
import math
import statistics


class EpisodeSetupProfile:
    """Sequential wall intervals, separate from the nested physical-step profile."""
    def __init__(self):
        self.started=self.last=perf_counter();self.rows=[]

    def mark(self,name):
        now=perf_counter();self.rows.append({'stage':name,'wall_s':now-self.last});self.last=now

    def record(self):
        return {'schema_version':'episode-setup-profile-v1','stages':list(self.rows),
            'total_s':sum(r['wall_s'] for r in self.rows),
            'scope':'Sequential wall intervals. GPU work is charged where completed. loop includes all per-step work. Not additive with nested step timings.'}


class EpisodeStepProfile:
    def __init__(self, max_steps, *, synchronize=False, fence=None):
        if type(max_steps) is not int or not 1 <= max_steps <= 2000:
            raise ValueError('Short profile requires 1..2000 committed steps')
        if type(synchronize) is not bool:
            raise ValueError('Profile synchronization must be Boolean')
        if synchronize and not callable(fence):
            raise ValueError('Synchronized profile requires an explicit device fence')
        self.max_steps=max_steps;self.synchronize=synchronize;self.fence=fence
        self.rows=[];self.current=None;self.last=None
        self.fence_calls=0;self.fence_wall_s=0.;self.pre_step_fence_s=0.

    def _fence(self):
        if self.synchronize:
            t=perf_counter();self.fence();self.fence_wall_s+=perf_counter()-t;self.fence_calls+=1

    def begin_step(self, step):
        if self.current is not None:raise RuntimeError('Unclosed profiling interval')
        t=perf_counter();self._fence();self.pre_step_fence_s+=perf_counter()-t
        self.last=perf_counter();self.current={'step':step,'stages_s':defaultdict(float)}

    def mark(self, name):
        if self.current is None:raise RuntimeError('No profiling interval')
        self._fence();now=perf_counter();elapsed=now-self.last
        if not math.isfinite(elapsed) or elapsed<0:raise RuntimeError('Invalid profiling clock')
        self.current['stages_s'][name]+=elapsed;self.last=now

    def end_step(self, phase):
        self.mark('progress_and_failure_check')
        self.current['phase']=phase
        self.current['total_s']=sum(self.current['stages_s'].values())
        self.current['stages_s']=dict(self.current['stages_s'])
        self.rows.append(self.current);self.current=None;self.last=None

    def record(self):
        def summarize(rows):
            names=sorted({name for r in rows for name in r['stages_s']})
            result={}
            for name in names:
                values=[r['stages_s'].get(name,0.) for r in rows];ordered=sorted(values)
                result[name]={'total_s':sum(values),'mean_ms':1000*statistics.mean(values),
                    'median_ms':1000*statistics.median(values),
                    'p95_ms':1000*ordered[max(0,math.ceil(.95*len(ordered))-1)]}
            return {'steps':len(rows),'total_s':sum(r['total_s'] for r in rows),'stages':result}
        warm=self.rows[100:] if len(self.rows)>100 else []
        return {'schema_version':'short-episode-step-profile-v1','requested_steps':self.max_steps,
            'completed_profile_steps':len(self.rows),'device_fenced':self.synchronize,
            'all_steps':summarize(self.rows),'after_first_100_steps':summarize(warm),
            'fence_calls':self.fence_calls,'fence_wall_s':self.fence_wall_s,
            'fence_time_included_in_stage_totals':True,'pre_step_fence_wall_s':self.pre_step_fence_s,
            'incomplete_step':None if self.current is None else self.current['step'],
            'incomplete_interval':None if self.current is None else {'step':self.current['step'],'stages_s':dict(self.current['stages_s'])},
            'scope':'Diagnostic wall attribution including device fences when enabled. Short prefix only, not delivery or carry certification.',
            'rows':self.rows}
