"""Instructor-owned vessel geometry shared by planning and physical assets.

No simulator dependency is imported. Fixed-density designs choose shape within
the bounds and a mass limit. The supplied code computes the component masses.
Component overlaps belong to the declared manufactured lumped-mass model.
"""

from .contracts import ContractError, number, point


CONTAINER_DENSITY_KG_M3 = 180.0


def build_container_geometry(container):
    """Return component solids, free interior regions and fixed-density inertia.

    component_boxes contains (size_xyz_m, center_xyz_m) pairs in payload-root
    coordinates. The hand is root + (0, 0, .105) under the upright down grasp.
    ``interior_regions`` is the union of disjoint free boxes. The older singular
    interior center/half-extents describe its enclosing box, which includes the
    solid divider when partitions=1. Use the regions for content accounting.
    This function provides geometry and physical mass. It does not certify
    graspability, retention or successful transport.
    """
    length, width, height = point(container["outer_size_m"])
    wall = number(container["wall_m"], "wall_m", minimum=0.003, maximum=0.025)
    handle_width = number(container["handle_width_m"], "handle_width_m", minimum=0.02, maximum=0.07)
    if not (0.06 <= length <= 0.24 and 0.06 <= width <= 0.24 and 0.03 <= height <= 0.18
            and min(length, width) > 2*wall and height > wall):
        raise ContractError("Container geometry is outside the provided component bounds")
    partitions = container.get("partitions")
    if container.get("open_top") is not True or type(partitions) is not int or partitions not in (0, 1):
        raise ContractError("The supplied vessel is open-top with zero or one partition")
    model = container.get("geometry_model", "offset_handle_v2")
    legacy = "empty_mass_kg" in container
    if legacy and "mass_limit_kg" in container:
        raise ContractError("Do not combine legacy chosen mass with the fixed-density mass limit")
    if model == "centered_overhead_handle_v3":
        if height > 0.06 or legacy:
            raise ContractError("Centered v3 requires height <= .06 m and a fixed-density mass limit")
        center_x, bottom = 0.0, -0.10
    elif model == "offset_handle_v2":
        if partitions:
            raise ContractError("A partition requires centered_overhead_handle_v3")
        center_x, bottom = 0.14, -0.02
    else:
        raise ContractError("Unknown provided container geometry_model")
    inner_height = height-wall
    center_z = bottom+wall+inner_height/2
    boxes = [
        ((length,width,wall), (center_x,0.,bottom+wall/2)),
        ((wall,width,inner_height), (center_x-length/2+wall/2,0.,center_z)),
        ((wall,width,inner_height), (center_x+length/2-wall/2,0.,center_z)),
        ((length-2*wall,wall,inner_height), (center_x,-width/2+wall/2,center_z)),
        ((length-2*wall,wall,inner_height), (center_x,width/2-wall/2,center_z)),
        ((.04,handle_width,.04), (0.,0.,0.)),
    ]
    if model == "centered_overhead_handle_v3":
        rim = bottom+height
        gap = -.031-rim
        boxes += [((length,.018,.012),(0.,0.,-.025)),
                  ((wall,.018,gap),(-length/2+wall/2,0.,(rim-.031)/2)),
                  ((wall,.018,gap),(length/2-wall/2,0.,(rim-.031)/2))]
    else:
        boxes.append(((.10,.025,.018),(.055,0.,-.011)))
    interior_half = [length/2-wall,width/2-wall,inner_height/2]
    if partitions:
        if width <= 3*wall:
            raise ContractError("The partition leaves no positive compartment width")
        # Full inner-height longitudinal X divider. It touches the floor and
        # end walls without adding overlapping solid volume at those joints.
        boxes.append(((length-2*wall,wall,inner_height),(center_x,0.,center_z)))
        half_y = (width-3*wall)/4
        center_y = (width-wall)/4
        regions = [{"id":name, "center_local_m":[center_x,sign*center_y,center_z],
                    "half_extents_m":[interior_half[0],half_y,interior_half[2]]}
                   for name, sign in (("negative_y",-1), ("positive_y",1))]
    else:
        regions = [{"id":"open", "center_local_m":[center_x,0.,center_z],
                    "half_extents_m":interior_half.copy()}]
    volume = sum(size[0]*size[1]*size[2] for size, _ in boxes)
    density = (number(container["empty_mass_kg"], "legacy empty mass", minimum=0.025, maximum=0.5)/volume
               if legacy else CONTAINER_DENSITY_KG_M3)
    mass = density*volume
    limit = (container["empty_mass_kg"] if legacy else
             number(container["mass_limit_kg"], "mass_limit_kg", minimum=0.025, maximum=0.5))
    if mass > limit + 1e-12:
        raise ContractError("Computed fixed-density container mass exceeds mass_limit_kg")
    component_masses = [density*size[0]*size[1]*size[2] for size, _ in boxes]
    component_mass_sum = sum(component_masses)
    com = [sum(m*pos[i] for m, (_,pos) in zip(component_masses,boxes))/component_mass_sum
           for i in range(3)]
    inertia = [[0.]*3 for _ in range(3)]
    for component_mass, (size,pos) in zip(component_masses,boxes):
        delta = [pos[i]-com[i] for i in range(3)]
        for i in range(3):
            other = [j for j in range(3) if j != i]
            inertia[i][i] += component_mass*(sum(size[j]**2 for j in other)/12
                                            +sum(delta[j]**2 for j in other))
            for j in range(i+1,3):
                inertia[i][j] -= component_mass*delta[i]*delta[j]
                inertia[j][i] = inertia[i][j]
    lower = [min(position[i]-size[i]/2 for size, position in boxes) for i in range(3)]
    upper = [max(position[i]+size[i]/2 for size, position in boxes) for i in range(3)]
    lower[2] -= .105
    upper[2] -= .105
    return {
        "geometry_model": model,
        "component_boxes": boxes,
        "interior_center_local_m": [center_x,0.,center_z],
        "interior_half_extents_m": interior_half,
        "interior_regions": regions,
        "interior_membership_rule": "union of free boxes; a whole sphere must fit within one region",
        "capacity_m3": sum(8*region["half_extents_m"][0]*region["half_extents_m"][1]
                           *region["half_extents_m"][2] for region in regions),
        "partitions": partitions,
        "payload_envelope_relative_to_hand_m": [lower, upper],
        "component_volume_m3": volume,
        "density_kg_m3": density,
        "empty_mass_kg": mass,
        "component_masses_kg": component_masses,
        "center_of_mass_local_m": com,
        "inertia_at_com_kg_m2": inertia,
        "mass_limit_kg": limit,
        "material_model_legacy": legacy,
        "material_policy_compatible": not legacy,
        "mass_model": "uniform density over supplied component boxes, overlaps counted once per component",
    }


def sphere_fits_interior(geometry, position_local_m, radius_m):
    """Test a complete sphere against free cavities, excluding solid dividers.

    A zero radius tests a particle center. This geometric predicate does not
    establish destination stability or approve any delivered quantity.
    """
    position = point(position_local_m)
    radius = number(radius_m, "radius_m", minimum=0.)
    return any(all(abs(position[axis]-region["center_local_m"][axis])+radius
                       <= region["half_extents_m"][axis]+1e-12 for axis in range(3))
               for region in geometry["interior_regions"])
