"""Unadopted standalone source callback, not a per-call SHA guarantee.

Usage: construct PinnedSourceGuard(pins), require before_request(), pass the
instance as source_check, always call after_request() in finally, then require
receipt()['request_complete_full_sha_verified']. Both request boundaries read
and SHA every path. Every active quick call stats every path. Metadata changes
or a due interval trigger a full SHA scan. No source mismatch can be reset.

ANY content change with the same observable size/mtime/ctime/ino/dev signature
can escape quick calls until full SHA. Causes include Windows timestamp
granularity/caching as well as intentional metadata spoofing. This is weaker
than hashing every call. Full initial/before/after and periodic checks remain
mandatory. The <=5s configured deadline is serviced at callbacks, without a
background thread or checks inside an opaque long oracle call. Only QA source
attestation frequency changes. No contact, physics or query predicate changes.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time


class PinnedSourceGuard:
    def __init__(self,pins,*,full_interval_s=5.,clock=time.monotonic):
        if not isinstance(pins,dict) or not pins:raise ValueError('Nonempty source pins required')
        if type(full_interval_s) not in (int,float) or not math.isfinite(full_interval_s) or not 0<full_interval_s<=5:
            raise ValueError('Full interval must be finite and in (0,5] seconds')
        if not callable(clock):raise TypeError('Clock must be callable')
        self._pins={}
        for name,expected in pins.items():
            path=Path(name).absolute()
            if (type(expected) is not str or len(expected)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected)
                    or path in self._pins):raise ValueError('Unique paths and exact SHA256 pins required')
            self._pins[path]=expected.lower()
        self._clock=clock;self._interval=float(full_interval_s);self._metadata={}
        self._ok=True;self._failure=None;self._active=False;self._complete=False
        self._started=0;self._finished=0;self._quick_calls=0;self._stat_calls=0
        self._full_checks=0;self._file_reads=0;self._full_reasons={};self._last_full=None;self._last_full_matched=False
        self._initial_verified=self._full('initial')

    def _fail(self,kind,path=None):
        if self._failure is None:self._failure={'kind':kind,'path':None if path is None else str(path)}
        self._ok=False;self._complete=False

    def _signature(self,path):
        self._stat_calls+=1
        s=path.stat()
        return (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)

    def _full(self,reason):
        self._full_checks+=1;self._full_reasons[reason]=self._full_reasons.get(reason,0)+1
        now=float(self._clock())
        if not math.isfinite(now) or (self._last_full is not None and now<self._last_full):
            self._fail('nonmonotonic_or_nonfinite_clock')
        self._last_full=now;matched=True;metadata={}
        for path,expected in self._pins.items():
            stable=False
            try:
                # A concurrent same-byte touch may settle on the second try.
                for _ in range(2):
                    before=self._signature(path)
                    actual=hashlib.sha256(path.read_bytes()).hexdigest();self._file_reads+=1
                    after=self._signature(path)
                    if actual!=expected:
                        self._fail('content_sha256_mismatch',path);matched=False;break
                    if before==after:
                        metadata[path]=after;stable=True;break
                if matched and not stable:
                    self._fail('unstable_metadata_during_full_sha',path);matched=False
            except OSError:
                self._fail('full_sha_read_or_stat_error',path);matched=False
        if len(metadata)!=len(self._pins):matched=False
        if matched:self._metadata=metadata
        self._last_full_matched=matched
        return self._ok and matched

    def checkpoint(self,label='manual'):
        """Always hash every path, even after a sticky failure."""
        if type(label) is not str or not label:raise ValueError('Checkpoint label required')
        return self._full('checkpoint:'+label)

    def before_request(self):
        if self._active:
            self._fail('request_already_active')
            return self._full('before_request')
        self._complete=False
        ok=self._full('before_request')
        if ok:self._active=True;self._started+=1
        return ok

    def after_request(self):
        active=self._active
        if not active:self._fail('missing_before_request_boundary')
        ok=self._full('after_request')
        self._active=False
        if active:self._finished+=1
        self._complete=bool(active and ok)
        return self._complete

    def __call__(self):
        self._quick_calls+=1
        if not self._active:self._fail('quick_check_outside_request')
        changed=False
        for path in self._pins:
            try:
                if self._signature(path)!=self._metadata.get(path):changed=True
            except OSError:
                self._fail('quick_stat_error',path);changed=True
        now=float(self._clock())
        if not math.isfinite(now) or now<self._last_full:
            self._fail('nonmonotonic_or_nonfinite_clock')
            return False
        if changed:return self._full('metadata_changed')
        if now-self._last_full>=self._interval:return self._full('periodic')
        return self._ok

    def receipt(self):
        return dict(schema_version='pinned-metadata-periodic-sha-source-guard-v1',pin_count=len(self._pins),
            initial_full_sha_verified=self._initial_verified,full_interval_s=self._interval,
            request_active=self._active,requests_started=self._started,requests_finished=self._finished,
            request_complete_full_sha_verified=bool(self._complete and self._ok and not self._active),
            sticky_ok=self._ok,failure=None if self._failure is None else dict(self._failure),
            quick_calls=self._quick_calls,stat_calls=self._stat_calls,full_checks=self._full_checks,
            full_file_reads=self._file_reads,full_reasons=dict(self._full_reasons),latest_full_sha_matched=self._last_full_matched,
            guarantee='Exact all-path SHA at initial/before/after/checkpoints and callback-serviced periodic deadlines. Every quick call stats all paths.',
            limitation='ANY content change with an unchanged observable stat signature can evade quick checks, including Windows timestamp granularity/caching and intentional spoofing. Full SHA boundaries/periodic checkpoints remain mandatory. No per-call cryptographic or background/mid-oracle guarantee. QA source-frequency tradeoff only, not a contact/physics/query gate change.')
