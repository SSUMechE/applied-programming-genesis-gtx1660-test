"""Supplied original native parameter evidence. No historical runner or route solver."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path

NOSLIP_SOURCES={
    "options/solvers.py":"2f799beaf1a5acb7aa55edcc063187f3e6a384ce1ea85092c2ee72f4e202e536",
    "engine/solvers/rigid/rigid_solver.py":"e076854af17262323041544ff692d366402a700357053618c8d89fed1e86b19f",
    "engine/solvers/rigid/constraint/solver.py":"f0753ff78bfe93b42d82ecb5c81e440cf0c521a3588efa3a4663a7c82e3b4d0f",
    "engine/solvers/rigid/constraint/noslip.py":"a3e39bf974d84f5bbfacb96511eed75766a3606302e3f011f4df8a166ed71ef2",
    "utils/array_class.py":"459ff6c67a3080d5ebf71e746d59f6746e6e76f11a1d534dfbfc7a78dd049d34",
}


TORSIONAL_SOURCES={**NOSLIP_SOURCES,
    "engine/materials/rigid.py":"4fca3e9d505e03d4496b2b8c4ef5a88a4cb6797e65f94c65f7da46b95a131704",
    "engine/entities/rigid_entity/rigid_geom.py":"399e0e0ec7d63a9a965c781f6946aed67109051ac8ccd666a47092cfb6a10a14",
    "engine/entities/rigid_entity/rigid_entity.py":"b0f86f9c9ebdf62b9df2e5aadc0719c28e06cd52e0ce1bef452b1a269d9b2092",
    "engine/solvers/rigid/collider/contact.py":"cd11feed70d9badfa9e56327e71cd49bcf9fe280bd8ed774b890ef6a1226023c",
    "utils/mjcf.py":"2606d714e2779a86d33ba696e7abce03a92ab3758276d6653f307f9240a6cee4",
    "utils/geom.py":"57affcc68874030ba0f0562516137d1a5c6373ae606037b2de34d4349a1499e3",
}


CONSTRAINT_PARAMETER_SOURCES={**TORSIONAL_SOURCES,
    "engine/entities/rigid_entity/rigid_equality.py":"2a2143af7d644dd2917947fffbadd6933bd384d2dc93ccdf0f5ae197f9e2b369",
    "engine/entities/rigid_entity/rigid_joint.py":"558c4224fff1b7cc991554278a9df75458186a61bdc08e0e2bb89c5ba21cfa84",
}


def verify_constraint_parameter_sources(installed):
    for name, expected in CONSTRAINT_PARAMETER_SOURCES.items():
        if sha(Path(installed)/name) != expected:
            raise ValueError("Installed constraint-parameter source changed: "+name)


def _parameter_vector(value, array):
    import numpy as np
    # joint_scene.array intentionally promotes to float64 for geometry math.
    # Preserve the native tensor dtype here so exact native-cast checks remain valid.
    raw=np.asarray(value.detach().cpu().numpy() if hasattr(value,'detach') else value)
    if raw.shape != (7,) or raw.dtype.kind != "f" or not np.isfinite(raw).all():
        raise ValueError("Expected seven finite floating constraint parameters")
    return raw.copy()


def _constraint_objects(entities):
    """Semantic ownership plus native IDs. No deduplication of compound geoms."""
    records=[]
    for entity in entities:
        joint_names={int(j.idx):j.name for j in entity.joints}
        for group, objects in (("geometries",entity.geoms),("joints",entity.joints),("equalities",entity.equalities)):
            for local_index,obj in enumerate(objects):
                if group=="geometries":
                    identity={"local_index":local_index,"link":obj.link.name,"type":int(obj.type)}
                elif group=="joints":
                    identity={"local_index":local_index,"name":obj.name,"type":int(obj.type)}
                else:
                    identity={"local_index":local_index,"type":int(obj.type),
                        "object_joint_names":[joint_names.get(int(obj.eq_obj1id)),joint_names.get(int(obj.eq_obj2id))],
                        "object_local_indices":[int(obj.eq_obj1id)-int(entity.joint_start),int(obj.eq_obj2id)-int(entity.joint_start)],
                        "eq_data":[float(v) for v in obj.eq_data]}
                records.append((group,obj,{"global_id":int(obj.idx),"entity":entity.name,"identity":identity}))
    return records


def native_bead_geometry(solver, beads, expected_radius, sphere_type, array):
    """Actual native primitive/radius, independent of the material ledger label."""
    import numpy as np
    if type(expected_radius) not in (int,float) or not math.isfinite(expected_radius) or expected_radius<=0:
        raise ValueError("Expected bead radius must be finite and positive")
    native_data=np.asarray(solver.dyn_info.geoms.data.to_numpy()).copy()
    native_type=np.asarray(solver.dyn_info.geoms.type.to_numpy()).copy()
    if native_data.shape!=(solver.n_geoms,7) or native_type.shape!=(solver.n_geoms,):
        raise ValueError("Native primitive field dimensions changed")
    rows=[]
    for bead in beads:
        if len(bead.geoms)!=1:raise ValueError("Each diagnostic bead must have one native sphere")
        geom=bead.geoms[0];data=native_data[int(geom.idx)]
        host=np.asarray(array(geom.data),dtype=data.dtype)
        if (geom.type!=sphere_type or native_type[int(geom.idx)]!=int(sphere_type)
                or host.shape!=(7,) or not np.array_equal(data,host) or not np.isfinite(data).all()
                or data[0]!=np.asarray(expected_radius,dtype=data.dtype).item() or np.any(data[1:]!=0)):
            raise ValueError("Built bead primitive or radius differs from fixed material supply")
        rows.append({"entity":bead.name,"geom_id":int(geom.idx),"native_type":int(geom.type),
            "native_type_name":"SPHERE","native_data":data.tolist(),"native_dtype":str(data.dtype),
            "radius_m":float(data[0]),"expected_radius_m":expected_radius})
    return rows


def tray_torsional_radius_evidence(ledger, requested):
    """Check an absolute material override and summarize possible pair coefficients."""
    import numpy as np
    requested=validate_tray_torsional_radius(requested)
    rows=ledger["geometries"]
    tray=[r for r in rows if r["entity"]=="tray_0"]
    if len(tray)!=9:raise ValueError("Tray material check requires all nine native vessel solids")
    dtype=np.dtype(ledger["native_coefficient_dtypes"]["friction_torsional"])
    if dtype.kind!="f":raise ValueError("Native torsional coefficient must use a floating dtype")
    for row in tray:
        public=row["public_coefficients"]["friction_torsional"]
        native=row["native_coefficients"]["friction_torsional"]
        if (type(public) not in (int,float) or type(native) not in (int,float)
                or not math.isfinite(public) or public<0 or not math.isfinite(native) or native<0
                or np.asarray(public,dtype=dtype).item()!=native):
            raise ValueError("Tray native torsional coefficient differs from public material value")
        if requested is not None and (public!=requested or native!=np.asarray(requested,dtype=dtype).item()):
            raise ValueError("Explicit tray torsional radius did not reach every native vessel solid")
    pairs=[]
    for name in sorted({r["entity"] for r in rows}-{ "tray_0" }):
        other=[r for r in rows if r["entity"]==name]
        values=[max(a["native_coefficients"]["friction_torsional"]*a["native_friction_ratio"],
                    b["native_coefficients"]["friction_torsional"]*b["native_friction_ratio"])
                for a in tray for b in other]
        if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError("Invalid native pair torsional coefficient")
        pairs.append({"other_entity":name,"minimum_m":min(values),"maximum_m":max(values)})
    return {"requested_radius_m":requested,"selection_mode":"parsed_geometry_or_engine_default" if requested is None else "absolute_tray_material_override",
        "native_dtype":str(dtype),"native_tray_values_m":[r["native_coefficients"]["friction_torsional"] for r in tray],
        "tray_geom_ids":[r["geom_id"] for r in tray],"possible_pair_coefficient_ranges_m":pairs,
        "pair_scope":"Initial coefficient combinations, not observed contacts or collision allowances.",
        "material_scope":"All nine tray solids. Other entity materials and rolling coefficients are unchanged.",
        "radius_basis":"At most1mm sensitivity parameter, below the smallest pad-face half-span3mm. Near-edge manifold witnesses do not establish a1mm physical disk.",
        "physical_patch_radius_calibrated":False,"teacher_physical_setting_approved":False}


def verify_torsional_sources(installed):
    for name,expected in TORSIONAL_SOURCES.items():
        if sha(Path(installed)/name)!=expected:
            raise ValueError(f"Installed torsional-friction implementation changed: {name}")
    return dict(TORSIONAL_SOURCES)


def torsional_configuration_evidence(solver, entities, requested, array, *, expected_rolling=False):
    """Read-only built flag/geometry observation. No contact moment is inferred."""
    import numpy as np
    requested=validate_torsional_friction(requested)
    expected_rolling=validate_torsional_friction(expected_rolling)
    if expected_rolling and not requested:
        raise ValueError("Native rolling friction requires torsional friction enabled")
    options=solver._options;config=solver.rigid_config
    for source in (options,config):
        if (type(source.enable_torsional_friction) is not bool or source.enable_torsional_friction!=requested
                or type(source.enable_rolling_friction) is not bool or source.enable_rolling_friction!=expected_rolling):
            raise ValueError("Torsional and rolling request/native flags must agree")
    if (solver.n_envs!=0 or config.enable_elliptic_friction is not False
            or type(config.rows_per_contact) is not int or config.rows_per_contact!=4+2*int(requested)+4*int(expected_rolling)):
        raise ValueError("Torsional diagnostic requires one unbatched pyramidal scene with exact native contact row count")
    count=solver.n_geoms
    if type(count) is not int or count<=0:raise ValueError("Need a nonempty built collision geometry set")
    names=("friction","friction_torsional","friction_rolling")
    native={name:np.asarray(getattr(solver.dyn_info.geoms,name).to_numpy()).copy() for name in names}
    ratio=np.asarray(array(solver.get_geoms_friction_ratio())).copy()
    for name,values in (*native.items(),("friction_ratio",ratio)):
        if values.shape!=(count,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all() or (values<0).any():
            raise ValueError(f"Need complete finite nonnegative native {name} coefficients")
    rows={}
    for entity in entities:
        for geom in entity.geoms:
            index=int(geom.idx)
            if index in rows or not 0<=index<count:raise ValueError("Duplicate or unknown native geometry ID")
            host={name:getattr(geom,name) for name in names}
            for name,value in host.items():
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value) or value<0:
                    raise ValueError(f"Invalid public geometry coefficient {name}")
                # Match the pinned engine's native-dtype cast, not a physical tolerance.
                if np.asarray(value,dtype=native[name].dtype).item()!=native[name][index]:
                    raise ValueError(f"Public/native geometry coefficient mismatch: {index}/{name}")
            rows[index]={"geom_id":index,"entity":entity.name,"entity_idx":int(entity.idx),
                "link":geom.link.name,"link_id":int(geom.link.idx),
                "public_coefficients":{name:float(value) for name,value in host.items()},
                "native_coefficients":{name:float(native[name][index]) for name in names},
                "native_friction_ratio":float(ratio[index])}
    if set(rows)!=set(range(count)):raise ValueError("Geometry friction ledger does not cover every built collision geometry")
    return {"requested_enable_torsional_friction":requested,
        "resolved_enable_torsional_friction":options.enable_torsional_friction,
        "native_enable_torsional_friction":config.enable_torsional_friction,
        "native_enable_rolling_friction":config.enable_rolling_friction,
        "native_rows_per_contact":config.rows_per_contact,"native_geom_count":count,
        "native_coefficient_dtypes":{name:str(values.dtype) for name,values in native.items()},
        "geometries":[rows[i] for i in range(count)],"source_sha256":dict(TORSIONAL_SOURCES),
        "coefficient_units":{"friction":"dimensionless","friction_torsional":"m effective patch radius","friction_rolling":"m"},
        "pair_rule":"max(native coefficient A * live friction_ratio A, native coefficient B * live friction_ratio B)",
        "observation_scope":"One-time built coefficients and initial friction ratios. No coefficient or ratio setters are called.",
        "contact_model_scope":"Global eligible contacts, including fingers, tray, beads, supports and map. Not a grip-only switch.",
        "direct_contact_spin_moment":"NOT_COLLECTED",
        "raw_contact_force_scope":"Linear force only. Position cross force omits the direct spin couple and is not total contact torque.",
        "teacher_physical_setting_approved":False}


def verify_noslip_sources(installed):
    for name,expected in NOSLIP_SOURCES.items():
        if sha(Path(installed)/name)!=expected:
            raise ValueError(f"Installed no-slip implementation changed: {name}")
    return dict(NOSLIP_SOURCES)


def solver_configuration_evidence(solver, requested, pyramidal_cone):
    """One-time host/options and built-scalar observation, without solving or stepping."""
    requested=validate_noslip_iterations(requested)
    options=solver._options
    keys=("constraint_solver","iterations","tolerance","ls_iterations","ls_tolerance",
          "noslip_iterations","noslip_tolerance","friction_cone","contact_resolution","impratio",
          "enable_torsional_friction","enable_rolling_friction","contact_pruning_tolerance",
          "enable_self_collision","enable_neutral_collision","enable_adjacent_collision","use_hibernation")
    resolved={key:getattr(options,key) for key in keys}
    for key,value in resolved.items():
        if value is not None and type(value) not in (bool,int,float,str):resolved[key]=str(value)
        if type(value) is float and not math.isfinite(value):raise ValueError(f"Nonfinite resolved option {key}")
    built={}
    for key in ("iterations","tolerance","ls_iterations","ls_tolerance","noslip_iterations","noslip_tolerance","impratio"):
        value=float(getattr(solver.rigid_info,key)[None])
        if not math.isfinite(value):raise ValueError(f"Nonfinite built solver setting {key}")
        if key.endswith("iterations"):
            if not value.is_integer() or value<0:raise ValueError(f"Invalid built iteration setting {key}")
            value=int(value)
        elif value<=0:raise ValueError(f"Nonpositive built solver setting {key}")
        built[key]=value
        if not math.isclose(value,getattr(options,key),rel_tol=1e-6,abs_tol=0.):
            raise ValueError(f"Built solver setting disagrees with resolved option {key}")
    if (type(options.noslip_iterations) is not int or options.noslip_iterations!=requested
            or options.friction_cone!=pyramidal_cone or solver._requires_grad or solver._use_hibernation):
        raise ValueError("No-slip diagnostic requires requested count, pyramidal friction, no differentiation or hibernation")
    return {"requested_noslip_iterations":requested,"resolved_rigid_options":resolved,"built_rigid_info":built,
            "requires_grad":bool(solver._requires_grad),"source_sha256":dict(NOSLIP_SOURCES),
            "observation_scope":"One-time configuration values. Iteration caps are not measured convergence counts.",
            "convergence_residual_history":"NOT_COLLECTED_NO_PUBLIC_PER_STEP_RESIDUAL_ADAPTER",
            "teacher_numerical_setting_approved":False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quaternion_error(a, b):
    from physics.carried_body import WorldPose
    qa=WorldPose((0,0,0),a).quaternion_wxyz
    qb=WorldPose((0,0,0),b).quaternion_wxyz
    return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(qa,qb)))))


def body_tilt_deg(quaternion):
    from physics.carried_body import WorldPose
    w,x,y,z=WorldPose((0,0,0),quaternion).quaternion_wxyz
    return math.degrees(math.acos(max(-1.,min(1.,1.-2.*(x*x+y*y)))))



NOSLIP_MAX_ITERATIONS=20  # Bounded instructor diagnostic, not a student setting.


TRAY_TORSIONAL_RADIUS_MAX_M=.001  # Contact-model sensitivity cap, not a calibrated patch radius.


def validate_torsional_friction(value):
    if type(value) is not bool:
        raise ValueError("torsional-friction must be an explicit Boolean")
    return value


def validate_tray_torsional_radius(value):
    if value is None:return None
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=TRAY_TORSIONAL_RADIUS_MAX_M:
        raise ValueError("tray-torsional-radius-m must be None or a finite number in 0..0.001m")
    return float(value)


def validate_noslip_iterations(value):
    if type(value) is not int or not 0<=value<=NOSLIP_MAX_ITERATIONS:
        raise ValueError(f"noslip-iterations must be an integer in 0..{NOSLIP_MAX_ITERATIONS}")
    return value

