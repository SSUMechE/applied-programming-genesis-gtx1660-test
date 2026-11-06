"""Particle-vessel contact adapter for the source-bound canonical nine-box union.

Caller supplies canonical_geometry only after canonical_reader_geometry and actual
native-component matching. This pure function checks consistency, not provenance.
Original rows/owners/component IDs survive. No contact is deleted or renamed as
solid_0. The existing classifier still decides every non-coplanar-bottom case.
"""
from __future__ import annotations

import numpy as np

from physics.episode_contacts import classify_episode_live_contacts
from physics.gravity_feedforward import rotation_matrix_wxyz
from physics.handle_contacts import MAX_PENETRATION_M, MINIMUM_NORMAL_COSINE, NUMERIC_TOLERANCE_M
from transport_challenge.contracts import ContractError, digest, point

REPRESENTATION_M = 1e-12  # Geometry-plane equality only. Existing contact tolerances are unchanged.
_WALLS = {'solid_1', 'solid_2', 'solid_3', 'solid_4'}


def _box(size, center):
    size, center = np.asarray(point(size, 3)), np.asarray(point(center, 3))
    if np.any(size <= 0):
        raise ContractError('Positive canonical/native box dimensions required')
    return size, center


def _axis_aligned(component):
    q = np.asarray(component.get('quaternion_body_wxyz', []), float)
    return bool(q.shape == (4,) and np.isfinite(q).all()
                and abs(abs(q[0])-1) <= REPRESENTATION_M and np.all(np.abs(q[1:]) <= REPRESENTATION_M))


def classify_particle_episode_live_contacts(raw, *, owners, components_by_geom, pads_by_arm,
        body_pose, finger_poses_by_arm, stations_by_geom, phase, active_holder_arm_id=None,
        canonical_geometry):
    """Same call/return contract as the original, plus explicit canonical geometry.

    Only a formerly UNALLOWED native wall-bottom/named-support-top witness can
    change category. It must lie on the same canonical floor-bottom plane and
    inside both actual boxes and the canonical floor footprint. The category
    describes the tray bottom. tray_component continues to name the actual wall.
    """
    rows, bilateral = classify_episode_live_contacts(raw, owners=owners,
        components_by_geom=components_by_geom, pads_by_arm=pads_by_arm, body_pose=body_pose,
        finger_poses_by_arm=finger_poses_by_arm, stations_by_geom=stations_by_geom,
        phase=phase, active_holder_arm_id=active_holder_arm_id)
    try:
        boxes = np.asarray(canonical_geometry['component_boxes'], float)
        if boxes.shape != (9, 2, 3) or not np.isfinite(boxes).all() or np.any(boxes[:, 0] <= 0):
            raise ContractError('Explicit canonical nine-box geometry required')
        by_name = {v['component_id']: v for v in components_by_geom.values()}
        if len(by_name) != 9 or set(by_name) != {'solid_'+str(i) for i in range(9)}:
            raise ContractError('Actual native nine component identities required')
        floor_size, floor_center = _box(*boxes[0].tolist()); floor = by_name['solid_0']
        native_size, native_center = _box(floor['size_m'], floor['center_body_m'])
        if (not _axis_aligned(floor) or np.max(np.abs(native_size-floor_size)) > REPRESENTATION_M
                or np.max(np.abs(native_center-floor_center)) > REPRESENTATION_M):
            raise ContractError('Canonical/native floor representation differs')
        bottom = float(floor_center[2]-floor_size[2]/2)
        rotation = rotation_matrix_wxyz(body_pose['quaternion_wxyz'])
        position = np.asarray(body_pose['position_m'])
        def inside(p, center, size):
            return bool(np.all(np.abs(p-center) <= size/2+NUMERIC_TOLERANCE_M))
        for row in rows:
            if row['category'] != 'UNALLOWED_CONTACT':
                continue
            pair = row['geom_ids']
            for tray_id, support_id in (pair, pair[::-1]):
                component, station = components_by_geom.get(tray_id), stations_by_geom.get(support_id)
                if (component is None or station is None or owners[tray_id]['kind'] != 'tray'
                        or owners[support_id]['kind'] != 'support' or component['component_id'] not in _WALLS
                        or not _axis_aligned(component)):
                    continue
                size, center = _box(component['size_m'], component['center_body_m'])
                if abs(float(center[2]-size[2]/2)-bottom) > REPRESENTATION_M:
                    continue
                p = np.asarray(row['position_world_m']); normal = np.asarray(row['normal_world'])
                local, body_normal = rotation.T@(p-position), rotation.T@normal
                support_local = p-np.asarray(station['center_world_m']); support_size = np.asarray(station['size_m'])
                depth = row['penetration_m']; surface = NUMERIC_TOLERANCE_M+max(0., depth)
                if (0. <= depth <= MAX_PENETRATION_M and row['normal_unit_error'] <= .001
                        and inside(local, center, size) and inside(support_local, np.zeros(3), support_size)
                        and np.all(np.abs(local[:2]-floor_center[:2]) <= floor_size[:2]/2+NUMERIC_TOLERANCE_M)
                        and abs(local[2]-bottom) <= surface
                        and abs(support_local[2]-support_size[2]/2) <= surface
                        and abs(body_normal[2]) >= MINIMUM_NORMAL_COSINE and abs(normal[2]) >= MINIMUM_NORMAL_COSINE):
                    row['category'] = 'named_tray_bottom_support_top'
                    row['particle_coplanar_bottom'] = dict(previous_category='UNALLOWED_CONTACT',
                        actual_component_id=component['component_id'], canonical_floor_bottom_body_m=bottom,
                        canonical_geometry_sha256=digest(canonical_geometry), representation_plane_tolerance_m=REPRESENTATION_M)
                break
        return rows, bilateral
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError('Malformed canonical particle contact geometry: '+str(exc)) from exc
