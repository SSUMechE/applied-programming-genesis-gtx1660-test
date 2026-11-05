"""Compact checked-step records with bounded, actual full failure context.

The runtime still owns every physical predicate, its recent rows and the first
exception. This writer never steps, restores or rechecks a scene. Completed rows
must be detached host trees whose physical values are not subsequently mutated.
The rigid16 runtime rebinds its sample list each interval. Keeping references to
those rows therefore adds a bounded window without copying their large trees.

Compact records preserve the complete outer observation and issued command.
They deliberately do not claim to retain every successful internal raw witness.
Full mode and failed-step records use the original full JSON conversion.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path


COMPACT_SCHEMA = 'compact-episode-step-v1'
FAILURE_SCHEMA = 'episode-failure-context-v1'
APPLICATION_NAME = 'Applied Programming'
OBSERVATION_KEYS = (
    'simulation_s', 'native_scene_step', 'arms', 'body_world',
    'body_linear_velocity_m_s', 'body_angular_velocity_rad_s',
    'body_up_tilt_deg', 'materials',
)


class CompactLogError(RuntimeError):
    """A logging failure must prevent successful episode acceptance."""


def _full_json(value):
    # Import only for full mode, an invalid compact value or failure publication.
    from physics.run_joint_path import _json_safe
    return json.dumps(_json_safe(value), ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)+'\n'


def _small_json(value):
    try:
        return json.dumps(value, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False)+'\n'
    except (TypeError, ValueError):
        # Preserve the existing explicit NaN/Inf markers. Do not sanitize the
        # much larger full source row on a normal successful write.
        return _full_json(value)


def compact_step_record(row):
    """Select actual outer data and counters, never fabricate a physical pass."""
    observation = {key: row[key] for key in OBSERVATION_KEYS}
    if 'provided_grasp' in row:
        observation['provided_grasp'] = row['provided_grasp']
    samples = row['rigid_substep_samples']
    if type(samples) is not list or len(samples) != 16:
        raise CompactLogError('Compact checked records require all sixteen actual samples')
    if row.get('post_processing_complete') is not True:
        raise CompactLogError('Compact records require the original completed checks')
    coverage = row['coverage']
    categories = Counter()
    for sample in samples:
        categories.update(item['category'] for item in sample['classified_contacts'])
    summary = dict(
        runtime_profile=row.get('rigid_profile'),
        observed_internal_samples=len(samples),
        first_global_substep=samples[0]['global_substep_ordinal'],
        last_global_substep=samples[-1]['global_substep_ordinal'],
        internal_native_contact_rows=sum(sample['contact_count'] for sample in samples),
        internal_classification_counts=dict(sorted(categories.items())),
        outer_native_contact_count=coverage['native_contact_count'],
        outer_copied_contact_count=coverage['copied_contact_count'],
        outer_classified_contact_count=coverage['classified_contact_count'],
        outer_contacts_complete=coverage['contacts_complete'],
        outer_policy_valid_now=coverage['policy_stamp']['policy_valid_now'],
        handling_passed=row.get('handling_interval', {}).get('passed'),
        handling_first_failure=row.get('handling_interval', {}).get('first_failure'),
    )
    return dict(schema_version=COMPACT_SCHEMA, application_name=APPLICATION_NAME,
        record_type='checked_step', step=row['step'], phase=row['phase'],
        command=row['command'], observation=observation, check_summary=summary,
        failures=row['failures'], post_processing_complete=True)


def decode_compact_video_frame(record):
    """Expose actual recorded video state without choosing a rate or interpolation."""
    if record.get('schema_version') != COMPACT_SCHEMA:
        raise CompactLogError('Expected a compact checked-step record')
    observation = record['observation']
    return dict(application_name=APPLICATION_NAME,
        step=observation['native_scene_step'], simulation_s=observation['simulation_s'],
        arms={name: dict(q=value['q_all_dofs'],
                       hand_position_m=value['hand_world']['position_m'])
              for name, value in observation['arms'].items()},
        body=observation['body_world'],
        materials=[dict(id=value['id'], entity_idx=value['entity_idx'],
            mass_kg=value['mass_kg'], position_m=value['position_world_m'],
            velocity_world_m_s=value['velocity_world_m_s'],
            inside_carrier=value['inside_carrier']) for value in observation['materials']],
        phase=record['phase'], active_arm=record['command']['active_arm_id'],
        action_role=record['command']['action_role'])


class CompactEpisodeLog:
    """Write synchronously. The caller retains ownership of the normal stream.

    ``write_step`` replaces the normal step encode/write/flush block only.
    ``record_failure`` belongs in the runtime's exception handler, before its
    pending samples are discarded. Catch a logging exception there as an extra
    cleanup error while preserving the original physical exception. ``finish``
    flushes but does not close the stream and cannot hide earlier I/O errors.
    """

    def __init__(self, stream, failure_context_path, *, mode='compact',
                 prefailure_steps=32, full_encoder=None):
        if mode not in ('compact', 'full'):
            raise CompactLogError('Choose explicit compact or full logging')
        if type(prefailure_steps) is not int or not 1 <= prefailure_steps <= 256:
            raise CompactLogError('The full-row reference window must contain 1 through 256 steps')
        if not callable(getattr(stream, 'write', None)) or not callable(getattr(stream, 'flush', None)):
            raise CompactLogError('An owned writable and flushable text stream is required')
        if full_encoder is not None and not callable(full_encoder):
            raise CompactLogError('Full encoder must be a callable')
        self.stream = stream
        self.failure_context_path = Path(failure_context_path).resolve()
        stream_name = getattr(stream, 'name', None)
        if isinstance(stream_name, (str, Path)) and Path(stream_name).resolve() == self.failure_context_path:
            raise CompactLogError('Failure context must use a separate file')
        self.mode = mode
        self._full_encoder = _full_json if full_encoder is None else full_encoder
        self._rows = deque(maxlen=prefailure_steps)
        self._written = self._compact = self._full = self._full_failure = 0
        self._internal_samples = self._characters = 0
        self._last_attempted = None
        self._first_failure = self._logging_error = None
        self._context_attempted = self._context_written = False
        self._context_error = self._context_sha256 = None
        self._saw_row_failure = self._finished = False

    def _error(self, exc):
        if self._logging_error is None:
            self._logging_error = dict(type=type(exc).__name__, message=str(exc))

    def _write_line(self, encoded):
        if type(encoded) is not str or not encoded.endswith('\n'):
            raise CompactLogError('The trusted encoder must return one newline-terminated JSON record')
        if self.stream.write(encoded) != len(encoded):
            raise CompactLogError('Short step-log write')
        self.stream.flush()

    def write_step(self, full_owned_row):
        """Keep no full copy or full-row digest on the compact normal path."""
        if self._finished or self._logging_error is not None:
            raise CompactLogError('A finished or failed writer cannot accept another step')
        self._last_attempted = full_owned_row
        try:
            if type(full_owned_row) is not dict:
                raise CompactLogError('A fully checked owned host row is required')
            step = full_owned_row.get('step')
            if type(step) is not int or step != self._written+1:
                raise CompactLogError('Completed step records must remain contiguous from one')
            if type(full_owned_row.get('failures')) is not list:
                raise CompactLogError('The original step failure list is required')
            row_failed = bool(full_owned_row['failures'])
            full = self.mode == 'full' or row_failed
            encoded = (self._full_encoder(full_owned_row) if full
                       else _small_json(compact_step_record(full_owned_row)))
            self._write_line(encoded)
            self._rows.append(full_owned_row)
            self._written += 1
            self._full += int(full)
            self._compact += int(not full)
            self._full_failure += int(row_failed)
            self._saw_row_failure |= row_failed
            self._internal_samples += len(full_owned_row.get('rigid_substep_samples', ()))
            self._characters += len(encoded)
            self._last_attempted = None
        except BaseException as exc:
            self._error(exc)
            raise CompactLogError('Episode step logging failed') from exc

    def record_failure(self, exc, *, pending=None, observation=None, stage=None):
        """Flush actual retained rows and partial pending state exactly once.

        No missing post-integration state is synthesized. A partial internal
        interval stays partial. Repeated calls never overwrite the first failure.
        """
        if self._first_failure is None:
            self._first_failure = dict(type=type(exc).__name__, message=str(exc))
        if self._context_written:
            return self.receipt()
        if self._context_attempted:
            raise CompactLogError('The first failure context could not be flushed')
        self._context_attempted = True
        context = dict(schema_version=FAILURE_SCHEMA, application_name=APPLICATION_NAME,
            first_failure=self._first_failure, stage=stage, mode=self.mode,
            last_flushed_step=self._written,
            retained_completed_step_rows=list(self._rows),
            partial_pending_step=pending, latest_observation=observation,
            unflushed_attempted_step_row=self._last_attempted,
            original_logging_error=self._logging_error,
            scope='Actual bounded owned host rows and partial pending samples only. No replay, inferred future state or additional physics.')
        try:
            encoded = self._full_encoder(context)
            if type(encoded) is not str or not encoded.endswith('\n'):
                raise CompactLogError('Failure encoder did not return a JSON record')
            with self.failure_context_path.open('x', encoding='utf-8', newline='\n') as stream:
                if stream.write(encoded) != len(encoded):
                    raise CompactLogError('Short failure-context write')
                stream.flush()
                os.fsync(stream.fileno())
            self._context_sha256 = hashlib.sha256(encoded.encode('utf-8')).hexdigest()
            self._context_written = True
        except BaseException as failure_exc:
            self._context_error = dict(type=type(failure_exc).__name__, message=str(failure_exc))
            self._error(failure_exc)
            raise CompactLogError('Actual failure context could not be flushed') from failure_exc
        return self.receipt()

    def finish(self):
        """Flush without closing. Earlier logging failures cannot become success."""
        if self._finished:
            return self.receipt()
        try:
            self.stream.flush()
            if self._logging_error is not None:
                raise CompactLogError('An earlier logging failure prevents successful finish')
            if self._saw_row_failure and not self._context_written:
                raise CompactLogError('A failed physical row has no flushed failure context')
        except BaseException as exc:
            self._error(exc)
            raise CompactLogError('Episode log final flush failed') from exc
        self._finished = True
        return self.receipt()

    def receipt(self):
        """A logging receipt is never a physical PASS or whole-mission verdict."""
        healthy = self._logging_error is None
        return dict(schema_version='episode-log-receipt-v1', application_name=APPLICATION_NAME,
            mode=self.mode, normal_record_schema=COMPACT_SCHEMA if self.mode == 'compact' else 'original-full-step',
            written_records=self._written, flushed_records=self._written,
            compact_records=self._compact, full_records=self._full,
            full_failed_step_records=self._full_failure,
            internal_samples_in_flushed_rows=self._internal_samples,
            serialized_characters=self._characters,
            prefailure_window_steps=self._rows.maxlen,
            retained_completed_steps=[row['step'] for row in self._rows],
            full_rows_copied_on_normal_write=0, measured_video_state_available=True,
            failure_recorded=self._first_failure is not None,
            first_failure=self._first_failure, failure_context_written=self._context_written,
            failure_context_path=str(self.failure_context_path),
            failure_context_sha256=self._context_sha256,
            failure_context_error=self._context_error, logging_error=self._logging_error,
            finished=self._finished,
            logging_passed=healthy and (self._first_failure is None or self._context_written),
            episode_success_eligible=self._finished and healthy and self._first_failure is None and not self._saw_row_failure,
            original_normal_internal_witnesses_retained=self.mode == 'full',
            scope='Logging health only. All physical predicates and recent-row acceptance remain owned by the original runtime. Compact normal records retain full outer observations, commands and internal counters, not every successful internal raw witness.')
