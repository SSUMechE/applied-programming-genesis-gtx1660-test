"""Course-created primitive geometry. No simulator import or work at import time."""
from pathlib import Path
import hashlib
import json
import sys
import math


def write_tray_urdf(directory, container=None):
    """One floating rigid link with separate solid boxes and an open interior.

    The grasp handle is the link origin. Every wall belongs to this one dynamic
    link. Unlike the October 25 visual probe, no wall is fixed to the world.
    The boxes have their own primitive collision shapes. Nothing fills the cup.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from transport_challenge.vessel import build_container_geometry
    except ModuleNotFoundError:
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
        from transport_challenge.vessel import build_container_geometry
    container=container or {"outer_size_m":[.16,.16,.102],"wall_m":.012,"open_top":True,"partitions":0,"handle_width_m":.04,"mass_limit_kg":.5}
    geometry=build_container_geometry(container)
    boxes=geometry["component_boxes"]
    asset_id=hashlib.sha256(json.dumps(container,sort_keys=True).encode()).hexdigest()[:12]
    path = directory / f"handled_open_tray_{asset_id}.urdf"
    # Uniform density over component boxes. Overlapping connector/handle volumes
    # are counted in this declared manufactured lumped-mass model.
    masses=geometry["component_masses_kg"]
    mass=sum(masses)
    center=geometry["center_of_mass_local_m"]
    inertia=geometry["inertia_at_com_kg_m2"]
    xyz = " ".join(map(str,center))
    parts = ['<?xml version="1.0"?>','<robot name="course_handled_open_tray"><link name="payload">',
             f'<inertial><origin xyz="{xyz}"/><mass value="{mass}"/><inertia ixx="{inertia[0][0]}" ixy="{inertia[0][1]}" ixz="{inertia[0][2]}" iyy="{inertia[1][1]}" iyz="{inertia[1][2]}" izz="{inertia[2][2]}"/></inertial>']
    for i,(size,pos) in enumerate(boxes):
        shape = " ".join(map(str,size)); origin = " ".join(map(str,pos))
        parts += [f'<collision name="solid_{i}"><origin xyz="{origin}"/><geometry><box size="{shape}"/></geometry></collision>',
                  f'<visual name="solid_{i}"><origin xyz="{origin}"/><geometry><box size="{shape}"/></geometry><material name="tray"><color rgba="0.55 0.58 0.62 1"/></material></visual>']
    parts += ['</link></robot>']
    content = "\n".join(parts)+"\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    contract={**geometry,"outer_size_m":container["outer_size_m"],"wall_m":container["wall_m"],"handle_width_m":container["handle_width_m"],
              "center_of_mass_local_m":center,"inertia_at_com_kg_m2":inertia,"open_top":True}
    return path, mass, contract


def make_payload(gs, scene, mission, source, assets, object_id="payload", container=None):
    if mission == "easy":
        size = (.04,.04,.04)
        body = scene.add_entity(gs.morphs.Box(size=size, pos=source),
             material=gs.materials.Rigid(friction=1.,rho=781.25), surface=gs.surfaces.Default(color=(.90,.34,.08)), name=object_id)
        return body, {"type":"single_dynamic_block","mass_kg":.05,"size_m":list(size),"object_reset_permitted":False}
    if mission == "rod":
        size = (.30,.035,.035)
        body = scene.add_entity(gs.morphs.Box(size=size,pos=source),
             material=gs.materials.Rigid(friction=1.,rho=.08/(.30*.035*.035)), surface=gs.surfaces.Default(color=(.90,.34,.08)), name=object_id)
        return body, {"type":"single_dynamic_rod","mass_kg":.08,"size_m":list(size),"object_reset_permitted":False,
                      "extraction_and_orientation_challenge":"NOT_IMPLEMENTED_IN_INITIAL_HANDOFF_PROBE"}
    path,mass,geometry = write_tray_urdf(assets,container)
    body = scene.add_entity(gs.morphs.URDF(file=str(path),pos=source,fixed=False,merge_fixed_links=True),
          material=gs.materials.Rigid(friction=1.,needs_coup=True,coup_friction=.4), name=object_id)
    return body,{"type":"single_dynamic_link_open_tray","mass_kg":mass,"asset":str(path),"asset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 **geometry,
                 "classification_note":"Union of actual free interior regions. Fluid/granular centers are representative points; rigid-ball accounting separately requires sphere clearance.",
                 "object_reset_permitted":False,"world_fixed_walls":False}


def ball_load_radius(load=None):
    """Case-owned teacher trial radius. Student designs never set material size."""
    if load is None:
        return .012
    radius=load.get("radius_m")
    if (load.get("type")!="fixed_ball_grid" or load.get("density_kg_m3")!=600.
            or isinstance(radius,bool) or not isinstance(radius,(int,float))
            or not math.isfinite(radius) or not .008<=radius<=.025):
        raise ValueError("Teacher fixed-ball load requires radius .008..025 m and fixed density 600 kg/m^3")
    return float(radius)


def ball_containment_tolerance(load=None):
    """Teacher measurement convention. Existing trials retain 1 mm tolerance."""
    value=.001 if load is None else load.get("containment_contact_tolerance_m",.001)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=.001:
        raise ValueError("Teacher sphere containment tolerance must be 0..0.001 m")
    return float(value)


def interior_membership(local_xyz,geometry,radius_m=0.,tolerance_m=0.):
    """One sphere/point must fit one free region; a solid divider is excluded."""
    import numpy as np
    points=np.asarray(local_xyz,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError("Interior membership needs finite N-by-3 local positions")
    if not math.isfinite(radius_m) or not math.isfinite(tolerance_m) or radius_m<0 or not 0<=tolerance_m<=radius_m:
        raise ValueError("Interior clearance requires nonnegative radius and bounded tolerance")
    regions=geometry.get("interior_regions") or [{"center_local_m":geometry["interior_center_local_m"],"half_extents_m":geometry["interior_half_extents_m"]}]
    inside=np.zeros(len(points),dtype=bool)
    for region in regions:
        center=np.asarray(region["center_local_m"],dtype=float)
        half=np.asarray(region["half_extents_m"],dtype=float)
        inside |= np.all(np.abs(points-center)<=half-(radius_m-tolerance_m),axis=1)
    return inside


def ball_load_positions(geometry, load=None):
    """Instructor fixed supply, distinct from the student's container design."""
    center=geometry["interior_center_local_m"]
    half=geometry["interior_half_extents_m"]
    radius=ball_load_radius(load)
    floor=center[2]-half[2]
    if load is None:
        points=[(center[0]+dx,center[1]+dy,floor+radius) for dx,dy in ((-.025,-.025),(-.025,.025),(.025,-.025),(.025,.025))]
        if not interior_membership(points,geometry,radius,tolerance_m=1e-12).all():
            raise ValueError("The fixed initial supply does not fit the student's open container")
        return points
    shape=load.get("grid_shape")
    if ("spacing_m" in load)==("spacing_xyz_m" in load):
        raise ValueError("Specify exactly one scalar spacing_m or three-axis spacing_xyz_m")
    spacing=load.get("spacing_xyz_m",[load.get("spacing_m")]*3)
    if not isinstance(shape,list) or len(shape)!=3 or any(type(v) is not int or not 1<=v<=8 for v in shape):
        raise ValueError("Ball grid shape must contain three bounded positive integers")
    if (not isinstance(spacing,list) or len(spacing)!=3 or any(isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or not 2*radius<=value<=.08 for value in spacing)):
        raise ValueError("Fixed initial balls must not overlap")
    ball_containment_tolerance(load)
    count=math.prod(shape)
    if count>96 or load.get("count")!=count:
        raise ValueError("Fixed finite ball count must equal the bounded grid product")
    points=[(center[0]+(i-(shape[0]-1)/2)*spacing[0],center[1]+(j-(shape[1]-1)/2)*spacing[1],floor+radius+k*spacing[2])
            for k in range(shape[2]) for j in range(shape[1]) for i in range(shape[0])]
    if not interior_membership(points,geometry,radius,tolerance_m=1e-12).all():
        raise ValueError("The fixed initial supply does not fit the student's open container")
    return points


def add_contents(gs, scene, mission, source, geometry=None,object_id="payload",load=None):
    if mission not in ("balls","water","sand"):
        return None,None
    x,y,z=source
    center=geometry["interior_center_local_m"] if geometry else [.14,0.,.037]
    half=geometry["interior_half_extents_m"] if geometry else [.068,.068,.045]
    inner_floor=center[2]-half[2]
    if mission == "balls":
        balls=[]
        radius=ball_load_radius(load)
        for index,point in enumerate(ball_load_positions(geometry,load)):
            balls.append(scene.add_entity(gs.morphs.Sphere(radius=radius,pos=(x+point[0],y+point[1],z+point[2])),
                material=gs.materials.Rigid(rho=600.,friction=.4), surface=gs.surfaces.Default(color=(.15+.17*(index%4),.45,.85-.15*(index%4))),name=f"{object_id}_ball_{index}"))
        return balls,{"type":"rigid_balls","initial_count":len(balls),"radius_m":radius,"density_kg_m3":600.,
                      "containment_contact_tolerance_m":ball_containment_tolerance(load),
                      "fixed_instructor_load":load,"particle_identity":"one persistent rigid entity per ID"}
    if load is not None:
        raise ValueError("This version's explicit material_load is for the fixed ball grid only")
    if geometry and geometry.get("partitions",0):
        raise ValueError("The partition prototype currently supports the explicit fixed-ball load only")
    material=gs.materials.SPH.Liquid() if mission=="water" else gs.materials.MPM.Sand()
    body=scene.add_entity(gs.morphs.Box(size=(.072,.072,.036),pos=(x+center[0],y+center[1],z+inner_floor+.002+.018)),material=material,
         surface=gs.surfaces.Default(color=(.12,.45,.90) if mission=="water" else (.78,.55,.22),vis_mode="particle"),name=f"{object_id}_contents")
    return body,{"type":"SPH.Liquid" if mission=="water" else "MPM.Sand","particle_size_m":.012,
                 "particle_mass_measurement":"version-pinned read-only actual solver per-particle mass adapter","reported_quantity":"retained simulated mass from moving-frame center classification",
                 "moving_container_coupling":"TO_BE_MEASURED","particle_reset_permitted":False}


def add_receiver(gs,scene,spec):
    """Fixed open receiving bin, distinct from the moving student container."""
    if spec is None:
        return [],None
    center=tuple(map(float,spec["center_m"]))
    length,width,height=map(float,spec["interior_size_m"])
    wall=float(spec["wall_m"])
    if not (.08<=length<=.4 and .08<=width<=.4 and .04<=height<=.25 and .004<=wall<=.03):
        raise ValueError("Receiver dimensions outside the supplied scene bounds")
    x,y,z=center
    lower=[x-length/2,y-width/2,z-height/2]
    upper=[x+length/2,y+width/2,z+height/2]
    shapes=[((length+2*wall,width+2*wall,wall),(x,y,lower[2]-wall/2)),
            ((wall,width+2*wall,height),(lower[0]-wall/2,y,z)),((wall,width+2*wall,height),(upper[0]+wall/2,y,z)),
            ((length,wall,height),(x,lower[1]-wall/2,z)),((length,wall,height),(x,upper[1]+wall/2,z))]
    bodies=[scene.add_entity(gs.morphs.Box(size=size,pos=pos,fixed=True),material=gs.materials.Rigid(needs_coup=True,coup_friction=.5),
              surface=gs.surfaces.Default(color=(.48,.48,.48)),name=f"receiver_wall_{i}") for i,(size,pos) in enumerate(shapes)]
    return bodies,{"interior_lower_world_m":lower,"interior_upper_world_m":upper,"fixed_open_bin":True,"specification":spec}
