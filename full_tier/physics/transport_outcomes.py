"""Classify recorded device checks without changing any physical predicate."""
from transport_challenge.contracts import ContractError

CONTROL_PHYSICAL = frozenset(('effort_saturation','position_limit','velocity_limit',
    'acceleration_limit','tracking_limit','tool_workspace'))
CONTROL_NUMERIC = frozenset(('nonfinite_model_or_gravity','quaternion_invalid','nonfinite_native_state',
    'motor_force_bound'))
CONTROL_INTEGRITY = frozenset(('model_changed','motor_identity','target_or_mode_changed'))
PAYLOAD_PHYSICAL = frozenset(('bead_outside','body_tilt'))
PAYLOAD_NUMERIC = frozenset(('nonfinite','quaternion_invalid'))
PAYLOAD_INTEGRITY = frozenset(('native_mass_changed',))
HANDLING_PHYSICAL = frozenset(('held_root_to_hand_exceeded','authenticated_hand_reach_exceeded',
    'authenticated_root_reach_exceeded','unheld_root_displacement_exceeded'))
HANDLING_NUMERIC = frozenset(('nonfinite_native_root_or_hand',))
HANDLING_INTEGRITY = frozenset(('contact_witness_failed','native_clock_or_pair_failed',
    'cup_native_mass_changed','direct_holder_switch','acquisition_without_named_fixed_support'))

class PhysicalTransportRejected(ContractError):
    """Finite native execution violated a supplied physical acceptance predicate."""

class DeviceSimulationError(ContractError):
    """The recorded execution cannot establish a valid physical rejection."""

def assess_device_checks(control, payload, contacts, handling):
    """A sticky physical failure remains a rejection. Numeric/integrity wins ties."""
    reasons=[]; physical=[]; numeric=[]; integrity=[]
    def add(group,area,name):
        item=area+':'+name
        reasons.append(item);group.append(item)
    for area,record,pp,nn,ii in (
            ('control',control,CONTROL_PHYSICAL,CONTROL_NUMERIC,CONTROL_INTEGRITY),
            ('payload',payload,PAYLOAD_PHYSICAL,PAYLOAD_NUMERIC,PAYLOAD_INTEGRITY)):
        if type(record) is not dict or type(record.get('flags')) is not dict:
            add(integrity,area,'missing_typed_checks');continue
        flags=record['flags']
        if set(flags)!=set(pp|nn|ii) or any(type(v) is not bool for v in flags.values()):
            add(integrity,area,'invalid_flag_contract');continue
        if record.get('source_bindings_unchanged') is not True:
            add(integrity,area,'source_changed')
        for name,value in flags.items():
            if value:add(physical if name in pp else numeric if name in nn else integrity,area,name)
        expected=not any(flags.values()) and record.get('source_bindings_unchanged') is True
        if record.get('passed') is not expected:add(integrity,area,'inconsistent_pass_flag')
    if type(contacts) is not dict or type(contacts.get('passed')) is not bool:
        add(integrity,'contacts','missing_typed_checks')
    else:
        code=contacts.get('first_bad_code')
        if contacts['passed'] is False and contacts.get('sticky_failed') is not True:
            add(integrity,'contacts','failed_without_sticky_witness')
        if type(code) is not int or code not in range(11):
            add(integrity,'contacts','unknown_failure_code')
        elif contacts['passed']:
            if code!=0 or contacts.get('sticky_failed') is not False:
                add(integrity,'contacts','inconsistent_pass_flag')
        elif code==9:
            add(physical,'contacts','unallowed_contact')
        elif code in (1,2,6,8):
            add(numeric,'contacts','native_error_'+str(code))
        else:
            add(integrity,'contacts','invalid_contact_evidence_'+str(code))
    if type(handling) is not dict or type(handling.get('passed')) is not bool:
        add(integrity,'handling','missing_typed_checks')
    else:
        if handling.get('source_bindings_verified') is not True:
            add(integrity,'handling','source_changed')
        first=handling.get('first_failure')
        if handling['passed'] is False and handling.get('sticky_failed') is not True:
            add(integrity,'handling','failed_without_sticky_witness')
        if handling['passed']:
            if first is not None or handling.get('sticky_failed') is not False:
                add(integrity,'handling','inconsistent_pass_flag')
        else:
            name=first.get('reason') if type(first) is dict else None
            if name=='contact_witness_failed' and type(contacts) is dict and contacts.get('passed') is False:
                pass  # The authenticated contact cause is classified above.
            elif name in HANDLING_PHYSICAL:add(physical,'handling',name)
            elif name in HANDLING_NUMERIC:add(numeric,'handling',name)
            else:add(integrity,'handling',name if name in HANDLING_INTEGRITY else 'unknown_failure')
    kind=('INTEGRITY_ERROR' if integrity else 'SIMULATION_ERROR' if numeric else
          'PHYSICAL_REJECTED' if physical else 'CHECKS_PASSED')
    return dict(schema_version='device-check-outcome-v1',outcome=kind,checks_passed=kind=='CHECKS_PASSED',
        physical_reasons=physical,numerical_reasons=numeric,integrity_reasons=integrity,
        failure_reasons=reasons,transport_complete=False,
        scope='Classification of supplied check receipts only. No new physics, acceptance threshold or complete transport claim.')

def reject_failed_checks(assessment):
    if assessment['checks_passed'] is not True:
        cls=PhysicalTransportRejected if assessment['outcome']=='PHYSICAL_REJECTED' else DeviceSimulationError
        raise cls(assessment['outcome']+': '+', '.join(assessment['failure_reasons']))

def review_outcome(*, accepted, timed_out, memory_limit_stop, physical_complete,
                   sources_current, diagnostic_outcomes):
    """Report a failed review distinctly without accepting incomplete transport."""
    if timed_out or memory_limit_stop:return 'RESOURCE_LIMIT'
    if not sources_current:return 'INTEGRITY_ERROR'
    values=[r.get('outcome') for r in diagnostic_outcomes]
    for kind in ('INTEGRITY_ERROR','SIMULATION_ERROR','PHYSICAL_REJECTED'):
        if kind in values:return kind
    if accepted:return 'ACCEPTED_TRANSPORT'
    if physical_complete:return 'RESOURCE_UNVERIFIED'
    return 'UNCLASSIFIED_FAILURE'

def json_safe_diagnostic(value):
    """Persist invalid numeric witnesses explicitly, never as JSON NaN tokens."""
    import math
    if isinstance(value,float) and not math.isfinite(value):return 'nonfinite:'+repr(value)
    if type(value) is dict:return {k:json_safe_diagnostic(v) for k,v in value.items()}
    if type(value) in (list,tuple):return [json_safe_diagnostic(v) for v in value]
    return value

def check_diagnostic(control,payload,contacts,handling):
    """Retain typed flags before a failure raises. Large arrays stay in NPZ."""
    checks={k:{n:v for n,v in record.items() if n!='history'} for k,record in
            (('control',control),('payload',payload),('contacts',contacts),('handling',handling))}
    return json_safe_diagnostic(dict(schema_version='device-check-diagnostic-v1',checks=checks,
        assessment=assess_device_checks(control,payload,contacts,handling)))

def read_device_diagnostics(output):
    """Read only genuine staged-check files and recompute their classification."""
    from pathlib import Path
    import json
    output=Path(output)/'device_execution'
    files=[output/'carry_checks.json',*sorted((output/'provided_tail').glob('phase_*_checks.json'))]
    rows=[]
    for p in files:
        if not p.is_file():continue
        value=json.loads(p.read_bytes())
        if value.get('schema_version')!='device-check-diagnostic-v1':
            raise ContractError('Invalid device failure diagnostic schema')
        c=value['checks'];actual=assess_device_checks(c['control'],c['payload'],c['contacts'],c['handling'])
        if actual!=value.get('assessment'):raise ContractError('Device diagnostic classification changed')
        rows.append(actual)
    return rows
