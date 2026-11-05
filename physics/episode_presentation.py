"""Explicit optional episode appearance, with unchanged vessel physics.

This pure adapter does not assign a physical material model. Water/sand colors
are presentation metadata until their actual physical instances are provided.
Promoted from the reviewed QA palette. Existing world records and measured
replays are untouched. Floor paint never defines collision or permissions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from physics.carried_body import VesselAsset, VesselModel
from transport_challenge.contracts import ContractError, digest


@dataclass(frozen=True)
class MaterialPalette:
    material_id: str
    label_ko: str
    body_hex: str
    rim_hex: str
    content_hex: str

    def to_record(self):
        return asdict(self)


_PALETTES = (
    MaterialPalette("balls", "공", "#8b4ddb", "#44205f", "#bd8cff"),
    MaterialPalette("water", "물", "#35bed6", "#075b99", "#54d5ee"),
    MaterialPalette("sand", "모래", "#b69a58", "#59402c", "#a97032"),
)


def material_palette(material_id: str) -> MaterialPalette:
    """Resolve an explicit material ID. No default or name guessing."""
    for item in _PALETTES:
        if type(material_id) is str and item.material_id == material_id:
            return item
    raise ContractError("Material presentation requires balls, water or sand")


def palette_record():
    return {
        "schema_version": "material-presentation-v2",
        "palette_version": 2,
        "supersedes_for_new_previews_only": "material-presentation-v1",
        "materials": [p.to_record() for p in _PALETTES],
        "legend_required": True,
        "physical_model_selected_by_color": False,
        "vessel_body_and_rim_distinguished": True,
    }


def rgb(value):
    if type(value) is not str or len(value) != 7 or not value.startswith("#"):
        raise ContractError("Expected an RGB hex color")
    try:
        return tuple(int(value[i:i+2], 16)/255. for i in (1, 3, 5))
    except ValueError as exc:
        raise ContractError("Invalid RGB hex color") from exc


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _physical_xml(root):
    tree = deepcopy(root)
    for link in tree.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    return ET.tostring(tree, encoding="utf-8")


def _vec(text):
    try:
        values = tuple(float(v) for v in text.split())
    except (AttributeError, ValueError) as exc:
        raise ContractError("Invalid visual box coordinates") from exc
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ContractError("Expected three finite visual box coordinates")
    return values


def _visual(name, center, size, hex_color):
    visual = ET.Element("visual", name=name)
    ET.SubElement(visual, "origin", xyz=" ".join(map(str, center)))
    ET.SubElement(ET.SubElement(visual, "geometry"), "box", size=" ".join(map(str, size)))
    ET.SubElement(ET.SubElement(visual, "material", name=name+"_color"), "color",
                  rgba=" ".join(map(str, (*rgb(hex_color), 1.))))
    return visual


def write_material_vessel_urdf(directory, source_asset: VesselAsset,
                               model: VesselModel, material_id: str, *,
                               rim_height_m: float = .003):
    """Return a new VesselAsset and provenance, changing only visual elements.

    Four existing wall visuals are split vertically into body and rim boxes.
    Their union equals the original visual box. No geometry protrudes and no
    collision solid, inertia, joint, physical material or body pose is added.
    Caller must use prioritize_urdf_material=True without a surface override.
    """
    palette = material_palette(material_id)
    if not isinstance(source_asset, VesselAsset) or not isinstance(model, VesselModel):
        raise ContractError("Require exact existing VesselAsset and VesselModel")
    if (type(rim_height_m) not in (int, float) or not math.isfinite(rim_height_m)
            or rim_height_m <= 0.):
        raise ContractError("Rim height must be a positive finite metre value")
    original = source_asset.path.read_bytes()
    if (_sha(original) != source_asset.asset_sha256
            or source_asset.model_geometry_sha256 != model.geometry_sha256
            or source_asset.empty_mass_kg != model.empty_mass_kg):
        raise ContractError("Source asset/model binding changed")
    from physics.loaded_joint_scene import validate_asset_components
    validate_asset_components(source_asset, model)
    root = ET.fromstring(original)
    links = root.findall("link")
    if len(links) != 1 or links[0].get("name") != "payload":
        raise ContractError("Expected the single authored payload link")
    link = links[0]
    visuals = link.findall("visual")
    expected_names = [f"solid_{i}" for i in range(len(model.component_boxes))]
    if [v.get("name") for v in visuals] != expected_names:
        raise ContractError("Unknown source visual/component mapping")
    if model.container_record().get("geometry_model", "offset_handle_v2") not in (
            "centered_overhead_handle_v3", "offset_handle_v2"):
        raise ContractError("Unreviewed vessel wall mapping")
    ledger = []
    before_physics = _physical_xml(root)
    for i, (visual, box) in enumerate(zip(visuals, model.component_boxes)):
        origin = visual.find("origin")
        geometry = visual.find("geometry/box")
        if (origin is None or geometry is None
                or origin.get("rpy", "0 0 0") != "0 0 0"
                or _vec(origin.get("xyz")) != box.center_body_m
                or _vec(geometry.get("size")) != box.size_m):
            raise ContractError("Source visual differs from exact component box")
        link.remove(visual)
        center, size = box.center_body_m, box.size_m
        pieces = []
        if 1 <= i <= 4:
            if rim_height_m >= size[2]:
                raise ContractError("Rim must leave a positive wall-body visual")
            body_size = (*size[:2], size[2]-rim_height_m)
            body_center = (*center[:2], center[2]-rim_height_m/2)
            rim_size = (*size[:2], float(rim_height_m))
            rim_center = (*center[:2], center[2]+(size[2]-rim_height_m)/2)
            pieces = [("body", body_center, body_size, palette.body_hex),
                      ("rim", rim_center, rim_size, palette.rim_hex)]
        else:
            role = "body" if i == 0 or i >= 9 else "handle"
            pieces = [(role, center, size, palette.body_hex if role == "body" else palette.rim_hex)]
        for role, xyz, dimensions, color in pieces:
            link.append(_visual(f"solid_{i}_{role}", xyz, dimensions, color))
        ledger.append({"collision_component_id": f"solid_{i}",
                       "original_visual_center_body_m": list(center),
                       "original_visual_size_m": list(size),
                       "visual_pieces": [{"role": role, "center_body_m": list(xyz),
                                          "size_m": list(dimensions), "color": color}
                                         for role, xyz, dimensions, color in pieces]})
    if _physical_xml(root) != before_physics:
        raise ContractError("Presentation unexpectedly changed nonvisual XML")
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)+b"\n"
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir/f"presentation_{material_id}_{_sha(data)[:16]}.urdf"
    if path.resolve() == source_asset.path.resolve():
        raise ContractError("Never overwrite the source vessel asset")
    if path.exists() and path.read_bytes() != data:
        raise ContractError("Conflicting existing presentation asset")
    path.write_bytes(data)
    if source_asset.path.read_bytes() != original:
        raise ContractError("Source vessel changed during presentation write")
    asset = VesselAsset(path.resolve(), _sha(data), model.geometry_sha256, model.empty_mass_kg)
    record = {"schema_version": "material-vessel-presentation-v1",
              "palette_version": 2, "palette_sha256": digest(palette_record()),
              "material": palette.to_record(), "rim_height_m": float(rim_height_m),
              "source_asset_sha256": source_asset.asset_sha256,
              "presentation_asset_sha256": asset.asset_sha256,
              "model_geometry_sha256": model.geometry_sha256,
              "nonvisual_xml_sha256": _sha(before_physics),
              "nonvisual_xml_unchanged": True, "components": ledger,
              "required_morph_options": {"align": False, "merge_fixed_links": True,
                                         "prioritize_urdf_material": True},
              "whole_entity_surface_override_permitted": False,
              "physical_material_or_supply_changed": False,
              "native_color_render_verified": False}
    record["presentation_sha256"] = digest(record)
    return asset, record


def shared_region_specs(world, visual):
    """All declared permissions, with separate source/goal display patches."""
    if 'mount_regions' not in world and 'relay_regions' not in world:return None
    mounts=world.get('mount_regions');relays=world.get('relay_regions')
    if not isinstance(mounts,list) or not mounts or not isinstance(relays,list) or not relays:
        raise ContractError('Shared mount and relay regions must both be explicit nonempty lists')
    specs=[];identities=set()
    def interval(values):
        if (not isinstance(values,(list,tuple)) or len(values)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in values)
                or values[0]>=values[1]):raise ContractError('Finite ordered display-region metre bounds required')
        return values
    def paint(name,x,y,color,role):
        if not isinstance(name,str) or not name or name in identities:raise ContractError('Unique display region identity required')
        identities.add(name);x=interval(x);y=interval(y);rgb(color)
        specs.append({'id':name,'size_m':[x[1]-x[0],y[1]-y[0],.0002],
            'center_world_m':[(x[0]+x[1])/2,(y[0]+y[1])/2,.0002],
            'color':color,'role':role,'collision':False})
    region_ids=[]
    for z in mounts:
        if not isinstance(z.get('id'),str) or not z['id']:raise ContractError('Named shared mount required')
        region_ids.append(z['id']);paint('display_base_region_'+z['id'],z['x_m'],z['y_m'],visual['mount_zone_color'],'shared_mount_region')
    for z in relays:
        if not isinstance(z.get('id'),str) or not z['id']:raise ContractError('Named relay region required')
        region_ids.append(z['id']);paint('display_relay_region_'+z['id'],z['center_x_m'],z['center_y_m'],visual['relay_color'],'relay_region')
    if len(region_ids)!=len(set(region_ids)):raise ContractError('Shared region identities alias')
    for role in ('source','goal'):
        s=world[role+'_support'];x,y,_=s['center_world_m'];sx,sy,_=s['size_m']
        paint('display_'+role,[x-sx/2,x+sx/2],[y-sy/2,y+sy/2],visual[role+'_color'],role)
    return specs


def board_line_specs(visual):
    """Exact reviewed preview-v5 edge/grid. Display-only, not a physical wall."""
    bounds=visual['floor_xy_m']
    if (not isinstance(bounds,list) or len(bounds)!=2 or any(not isinstance(b,list) or len(b)!=2
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in b) or b[0]>=b[1] for b in bounds)):
        raise ContractError('Finite ordered display floor bounds required')
    (x0,x1),(y0,y1)=bounds;lines=[]
    def line(name,xlo,xhi,ylo,yhi,color,role):
        lines.append({'id':name,'size_m':[xhi-xlo,yhi-ylo,.00005],
            'center_world_m':[(xlo+xhi)/2,(ylo+yhi)/2,.00006],
            'color':color,'role':role,'collision':False,
            'meaning':'Display-only scale/edge. Not a wall, hazard, placement permission or task constraint.'})
    width=.008
    if min(x1-x0,y1-y0)<=2*width:raise ContractError('Board bounds too small for display outline')
    line('display_board_south',x0,x1,y0,y0+width,'#b8b8b8','board_outline')
    line('display_board_north',x0,x1,y1-width,y1,'#b8b8b8','board_outline')
    line('display_board_west',x0,x0+width,y0,y1,'#b8b8b8','board_outline')
    line('display_board_east',x1-width,x1,y0,y1,'#b8b8b8','board_outline')
    spacing=.25;half=.001
    for axis,(lo,hi) in enumerate(bounds):
        ticks=range(math.floor(lo/spacing)+1,math.ceil(hi/spacing))
        if len(ticks)>200:raise ContractError('Unbounded display grid request')
        for ordinal,tick in enumerate(ticks):
            p=tick*spacing
            if not lo+width+half<p<hi-width-half:continue
            if axis==0:line(f'display_grid_x_{ordinal}',p-half,p+half,y0+width,y1-width,'#dedede','quarter_metre_grid')
            else:line(f'display_grid_y_{ordinal}',x0+width,x1-width,p-half,p+half,'#dedede','quarter_metre_grid')
    return lines


def framed_camera_options(camera):
    expected,replacement={'overview':(47,32),'west_overview':(48,33),'top_oblique':(43,40)}[camera['name']]
    if camera['fov_deg']!=expected:raise ContractError('Board framing expects the reviewed camera FOV')
    delta=[b-a for a,b in zip(camera['position_m'],camera['lookat_m'])]
    up=camera.get('up',[0,1,0] if delta[0]==delta[1]==0 else [0,0,1])
    if len(up)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in up):raise ContractError('Invalid camera up vector')
    cross=[delta[1]*up[2]-delta[2]*up[1],delta[2]*up[0]-delta[0]*up[2],delta[0]*up[1]-delta[1]*up[0]]
    if sum(v*v for v in cross)==0:raise ContractError('Camera up is parallel to view')
    return {'pos':deepcopy(camera['position_m']),'lookat':deepcopy(camera['lookat_m']),'fov':replacement,'up':deepcopy(up)}
