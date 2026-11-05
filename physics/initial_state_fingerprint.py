"""One boundary read of fresh native INITIAL observables, not a checkpoint.

No simulator is imported here. Missing native fields fail comparison closed.
Solver/coupler/RNG state restoration remains explicitly unqualified.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np

SCHEMA='fresh-native-initial-observables-v1'
MISSING_INTERNAL=('complete_native_solver_scratch_and_warmstart_reset_proof',
    'complete_coupler_and_collision_cache_restore_proof','native_and_host_RNG_restore_proof',
    'application_controller_grasp_dispatcher_history_restore_proof',
    'fresh_vs_restored_same_command_differential_qualification')
RIGID_FIELDS={
 'rigid_info.qpos':('q',None),
 'dyn_state.dofs.pos':('d',None),'dyn_state.dofs.vel':('d',None),
 'dyn_state.dofs.acc':('d',None),'dyn_state.dofs.vel_prev':('d',None),
 'dyn_state.dofs.vel_next':('d',None),'dyn_state.dofs.ctrl_pos':('d',None),
 'dyn_state.dofs.ctrl_vel':('d',None),'dyn_state.dofs.ctrl_force':('d',None),
 'dyn_state.dofs.ctrl_mode':('d','integer'),
 'dyn_state.links.pos':('l',3),'dyn_state.links.quat':('l',4),
 'dyn_state.links.cd_vel':('l',3),'dyn_state.links.cd_ang':('l',3),
 'dyn_state.links.root_COM':('l',3),'dyn_state.links.i_pos_shift':('l',3),
 'dyn_state.links.mass_shift':('l',None),
 'dyn_state.links.cfrc_applied_vel':('l',3),'dyn_state.links.cfrc_applied_ang':('l',3),
 'dyn_info.dofs.act_gain':('di',None),'dyn_info.dofs.act_bias':('di',3),
 'dyn_info.dofs.force_range':('di',2),'dyn_info.dofs.limit':('di',2),
 'dyn_info.links.inertial_mass':('li',None),'dyn_info.links.inertial_pos':('li',3),
 'dyn_info.links.inertial_quat':('li',4),
 'rigid_info.substep_dt':('scalar',None),'rigid_info.gravity':('one',3),
 'rigid_info.tolerance':('scalar',None),'rigid_info.iterations':('scalar','integer'),
 '_errno':('one','integer'),
 'constraint_solver.constraint_state.qd_n_equalities':('one','integer')}


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _get(obj,path):
    for part in path.split('.'):obj=getattr(obj,part)
    return obj


def required_native_fields(kind):
    result=['rigid.'+name for name in RIGID_FIELDS]
    result+=['active_equalities.'+name for name in ('eq_type','eq_obj1id','eq_obj2id','eq_data','sol_params')]
    if kind in ('process_liquid','sand'):
        result+=['particles.'+name for name in ('pos','vel','active')]
        if kind=='sand':result+=['particles.'+name for name in ('C','F','Jp')]
        info=('mass','rho','mu','gamma','stiffness','exponent') if kind=='process_liquid' else ('mass','material_idx','free','default_Jp')
        result+=['particles_info.'+name for name in info]
    return result


def array_record(value):
    """Retain dtype/shape and exact bytes. Numeric data is never JSON-rounded."""
    a=np.asarray(value)
    if a.dtype.kind not in 'biuf':raise ValueError('Native numeric array required')
    a=np.ascontiguousarray(a)
    return dict(dtype=a.dtype.str,shape=list(a.shape),nbytes=a.nbytes,
        sha256=hashlib.sha256(a.tobytes()).hexdigest(),finite=bool(np.isfinite(a).all()))


def native_array(value, converter=None):
    """One owned host read through the installed native converter at INITIAL."""
    if hasattr(value,'detach') and hasattr(value,'cpu'):
        return np.asarray(value.detach().cpu().numpy()).copy()
    if converter is None:
        from genesis.utils.misc import qd_to_numpy
        converter=qd_to_numpy
    return np.asarray(converter(value,copy=True)).copy()


def capture_initial_fingerprint(context):
    """Call after controls/helper/INITIAL setup and before the first scene.step.

    Required context: scene, arms, tray, beads, ids, model, world, scenario,
    settings, configuration, automatic_grasp, handling_enforcer, dispatcher,
    initial_observation, and particle_reader (None for beads). No setter runs.
    """
    out=dict(schema_version=SCHEMA,status='MISSING_REQUIRED_INITIAL_FIELDS',
        capture_scope='FRESH_SCENE_CLOCK_ZERO_NATIVE_PHYSICAL_AND_CONTROL_OBSERVABLES',
        missing_required_fields=[],fields={},fresh_scene_comparison_eligible=False,
        restored_state_claimed=False,full_checkpoint_inventory_complete=False,
        checkpoint_reuse_qualified=False,missing_internal_checkpoint_fields=list(MISSING_INTERNAL),
        source_files={},capture_phase='after_initial_setup_before_first_physics_step')
    out['declared_native_inventory']={kind:required_native_fields(kind) for kind in ('beads','process_liquid','sand')}
    missing=out['missing_required_fields'];fields=out['fields']
    def field(name,obj,shape=None,allow_nonfinite=False):
        try:
            value=native_array(obj)
            a=np.asarray(value)
            if shape is not None and a.shape!=tuple(shape):raise ValueError('Native shape differs: '+str(a.shape))
            item=array_record(a)
            if not allow_nonfinite and not item['finite']:raise ValueError('Nonfinite initial state')
            fields[name]=item
            return a.copy()
        except Exception as exc:
            missing.append(dict(field=name,reason=type(exc).__name__+': '+str(exc)))
            return None
    try:
        scene=context['scene'];r=scene.rigid_solver
        if (type(scene).__module__!='genesis.engine.scene'
                or not type(r).__module__.startswith('genesis.engine.solvers.rigid')):
            raise ValueError('Actual installed Genesis scene/rigid owner required, not caller arrays')
        clock=[int(scene.t),float(scene.cur_t),int(scene.sim.cur_substep_global),int(scene.sim.cur_substep_local)]
        out['native_clock']=clock
        if clock!=[0,0.,0,0]:raise ValueError('Only actual fresh native clock zero is admitted')
        if scene.n_envs!=0 or r._B!=1:raise ValueError('Single native environment required')
        for path,(kind,width) in RIGID_FIELDS.items():
            counts={'q':r.n_qs,'d':r.n_dofs,'l':r.n_links,
                    'di':r.n_dofs,'li':r.n_links}
            if kind in ('scalar','one'):shape=() if kind=='scalar' else (1,)
            else:
                batched=kind in ('q','d','l') or (kind=='di' and r.rigid_config.batch_dofs_info) or (kind=='li' and r.rigid_config.batch_links_info)
                shape=(counts[kind],1) if batched else (counts[kind],)
            if type(width) is int:shape=(*shape,width)
            try:obj=_get(r,path)
            except Exception as exc:
                missing.append(dict(field='rigid.'+path,reason=type(exc).__name__+': '+str(exc)));continue
            field('rigid.'+path,obj,shape,allow_nonfinite=path in ('dyn_info.dofs.force_range','dyn_info.dofs.limit'))
        qd=fields.get('rigid.rigid_info.qpos',{}).get('dtype')
        for name,item in fields.items():
            if 'ctrl_mode' not in name and 'qd_n_equalities' not in name and '_errno' not in name and 'iterations' not in name:
                if item['dtype']!=qd:missing.append(dict(field=name,reason='Mixed unexpected native floating dtype'))
        from physics.live_contact_coverage import automatic_grasp_witness
        proof=automatic_grasp_witness(context['automatic_grasp'],scene=scene)
        if proof['holder_arm_id'] is not None:raise ValueError('Fresh INITIAL already has a holder')
        inventory=proof['snapshot']['inventory'];n=inventory['count']
        for name in ('eq_type','eq_obj1id','eq_obj2id','eq_data','sol_params'):
            raw=native_array(_get(r,'dyn_info.equalities.'+name))
            a=np.asarray(raw)
            if a.shape[:2]!=(inventory['capacity'],1):raise ValueError('Native equality ABI differs')
            fields['active_equalities.'+name]=array_record(a[:n])
        out['grasp_baseline']=dict(holder=None,owned_weld_row=proof['owned_weld_row'],
            baseline=proof['baseline_inventory'],actual_inventory=inventory,
            cup_link_idx=proof['snapshot']['cup_link_idx'],body_entity_idx=proof['snapshot']['body_entity_idx'],
            hand_links_by_arm=proof['hand_links_by_arm'],public_cup_mass_kg=proof['snapshot']['cup_mass_kg'])
        guard=context['handling_enforcer'].guard.status()
        dispatch=context['dispatcher'].record()
        if guard['last_accepted_scene_step']!=0 or guard['observations_accepted']!=0 or guard['qualified_holder_id'] is not None:
            raise ValueError('Application handling history is not fresh')
        out['application_initial']=dict(handling_guard=guard,
            dispatcher_global_step=dispatch['global_step'],dispatcher_simulation_s=dispatch['simulation_s'])
        arms=context['arms'];tray=context['tray'];beads=context['beads'];ids=context['ids']
        out['native_identity']=dict(arms={name:dict(entity_idx=int(arm.idx),base_link_idx=int(arm.base_link_idx),
            dof_start=int(arm.dof_start),n_dofs=int(arm.n_dofs)) for name,arm in sorted(arms.items())},
            cup=dict(entity_idx=int(tray.idx),base_link_idx=int(tray.base_link_idx)),
            beads=[dict(id=item['id'],entity_idx=int(bead.idx),link_idx=int(bead.base_link_idx),
                        logical_mass_kg=item['mass_kg'],radius_m=item['radius_m']) for bead,item in zip(beads,ids)])
        if len(beads)!=len(ids):raise ValueError('Bead identities were omitted')
        reader=context.get('particle_reader')
        kind='beads' if reader is None else reader.kind
        out['material_kind']=kind
        if reader is None:
            if len(ids)!=4:raise ValueError('Original four bead identities required')
        else:
            reader.verify_source_bindings();reader._topology()
            s=reader.solver;start,end=reader.start,reader.end;count=reader.n
            if start!=0 or end!=s.n_particles or count!=s.n_particles:raise ValueError('Whole original particle inventory required')
            out['particle_identity']=dict(contract_sha256=reader.contract_sha256,
                native_particle_ids=list(range(start,end)),count=count,source_id=reader.source_id)
            frame=int(scene.sim.cur_substep_local)
            for name in ('pos','vel')+(() if kind=='process_liquid' else ('C','F','Jp')):
                raw=native_array(getattr(s.particles,name))
                expected=(count,1) if kind=='process_liquid' else (scene.sim.substeps_local+1,count,1)
                if raw.shape[:len(expected)]!=expected:raise ValueError('Particle ring/ID ABI differs')
                values=raw if kind=='process_liquid' else raw[frame]
                fields['particles.'+name]=array_record(values)
            raw=native_array(s.particles_ng.active)
            expected=(count,1) if kind=='process_liquid' else (scene.sim.substeps_local+1,count,1)
            if raw.shape!=expected:raise ValueError('Particle active ABI differs')
            fields['particles.active']=array_record(raw if kind=='process_liquid' else raw[frame])
            info=('mass','rho','mu','gamma','stiffness','exponent') if kind=='process_liquid' else ('mass','material_idx','free','default_Jp')
            for name in info:field('particles_info.'+name,getattr(s.particles_info,name),(count,))
        world=context['world'];settings=context['settings'];configuration=context['configuration']
        out['model_signature']=digest(dict(world_sha256=digest(world),settings_sha256=digest(settings),
            native_configuration_sha256=digest(configuration),geometry_sha256=context['model'].geometry_sha256,
            material_kind=kind,clock=dict(outer_dt_s=float(scene.sim.dt),substeps=int(scene.sim.substeps),
                local_substeps=int(scene.sim.substeps_local))))
        out['initial_observation_sha256']=digest(context['initial_observation'])
        for obj in (type(scene),type(scene.sim),type(r),type(r.dyn_state)):
            p=Path(inspect.getfile(obj)).resolve();out['source_files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if clock!=[int(scene.t),float(scene.cur_t),int(scene.sim.cur_substep_global),int(scene.sim.cur_substep_local)]:
            raise ValueError('Native clock changed during INITIAL capture')
    except Exception as exc:missing.append(dict(field='native_context',reason=type(exc).__name__+': '+str(exc)))
    if not missing:
        expected=required_native_fields(out['material_kind'])
        missing.extend(dict(field=name,reason='Required inventory field was not captured') for name in expected if name not in fields)
        missing.extend(dict(field=name,reason='Nonfinite initial physical/constraint data') for name,item in fields.items()
                       if not item['finite'] and name not in ('rigid.dyn_info.dofs.force_range','rigid.dyn_info.dofs.limit'))
        if not missing:
            out['status']='FRESH_INITIAL_OBSERVABLES_CAPTURED';out['fresh_scene_comparison_eligible']=True
    out['fingerprint_sha256']=digest({k:out[k] for k in
        ('model_signature','native_identity','fields','grasp_baseline','application_initial') if k in out})
    out['record_sha256']=digest(out)
    return out


def compare_initial_fingerprints(reference,current):
    failures=[]
    for label,record in (('reference',reference),('current',current)):
        if (record.get('schema_version')!=SCHEMA or record.get('status')!='FRESH_INITIAL_OBSERVABLES_CAPTURED'
                or record.get('missing_required_fields') or record.get('fresh_scene_comparison_eligible') is not True
                or not set(required_native_fields(record.get('material_kind'))).issubset(record.get('fields',{}))
                or record.get('record_sha256')!=digest({k:v for k,v in record.items() if k!='record_sha256'})):
            failures.append(label+': incomplete/changed initial fingerprint')
    if not failures:
        for key in ('model_signature','native_identity','fields','grasp_baseline','application_initial'):
            if reference.get(key)!=current.get(key):failures.append('Different '+key)
    return dict(schema_version='fresh-initial-observable-comparison-v1',passed=not failures,failures=failures,
        reference_sha256=reference.get('record_sha256'),current_sha256=current.get('record_sha256'),
        comparison_scope='Exact fresh scene physical/control observable fields only',
        full_state_restoration_verified=False,checkpoint_reuse_qualified=False)
