"""Provided unloaded query guard preserved from the reviewed source approach.

This is extra conservative planning clearance, never a physical contact
allowance. The exact helper body is reused from the earlier QA implementation.
"""
import numpy as np


def install_open_finger_margin(query,margin_m=.003):
    """Additional conservative planner clearance, never a physical allowance.

    Public world AABBs for every active finger collision geom versus every
    vessel component. Only unloaded/open-finger query scenes use this guard.
    RRT still generates every intermediate q. No grasp route is inserted.
    """
    if hasattr(query,'held_body') or hasattr(query,'open_finger_margin_m'):return
    from physics.joint_scene import array
    fingers=[g for g in query.scene.rigid_solver.geoms if query._owners[int(g.idx)].get('kind')=='arm'
        and query._owners[int(g.idx)].get('entity')==query.active_arm_id
        and query._owners[int(g.idx)].get('link') in ('left_finger','right_finger')]
    body=[g for g in query.scene.rigid_solver.geoms if query._owners[int(g.idx)].get('kind')=='tray']
    assert fingers and body
    bb=[array(g.get_AABB()) for g in body]
    original=query.state_valid;query.open_finger_margin_m=margin_m
    def valid(q):
        if not original(q):return False
        for finger in fingers:
            aa=array(finger.get_AABB())
            for box in bb:
                gap=np.maximum(np.maximum(aa[0]-box[1],box[0]-aa[1]),0.)
                if float(np.linalg.norm(gap))<margin_m:
                    return query._reject('open_finger_body_aabb_clearance',minimum_m=margin_m)
        return True
    query.state_valid=valid
