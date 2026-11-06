"""Sparse raw Genesis frames. No simulation step, pose setter or image editing."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from transport_challenge.contracts import ContractError,digest

SPARSE_CHECKPOINTS=('lifted','relay_release','recipient_regrasp','unit_lifted','unit_released',
    'mission_arm_0_relay_release','mission_arm_1_relay_release','mission_arm_2_goal_release',
    'empty_grasp_start','empty_path_end')


class SparseNativeImages:
    def __init__(self,scene,camera,output,observe):
        self.scene,self.camera,self.output,self.observe=scene,camera,Path(output),observe
        self.records=[]

    def capture(self,checkpoint,observation,detail):
        if checkpoint not in SPARSE_CHECKPOINTS:
            raise ContractError('Unknown sparse checkpoint')
        if any(r['checkpoint']==checkpoint for r in self.records):raise ContractError('Checkpoint already captured')
        from PIL import Image
        def vector(value):
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            value=np.asarray(value,dtype=float).copy()
            if value.shape!=(3,) or not np.isfinite(value).all():raise ContractError('Camera/body vectors must be finite XYZ')
            return value
        body=vector(observation['body_world']['position_m'])
        saved={name:vector(getattr(self.camera,'get_'+name)()) for name in ('pos','lookat','up')}
        before=self.observe();clock=(int(self.scene.t),float(self.scene.cur_t));entries=[]
        material_keys=(('particle_material','particle_pause_state') if 'particle_material' in observation else ('materials',))
        for key in ('simulation_s','native_scene_step','body_world',*material_keys):
            if before[key]!=observation[key]:raise ContractError('Image checkpoint observation is stale')
        for name,arm in before['arms'].items():
            if any(arm[k]!=observation['arms'][name][k] for k in ('q_all_dofs','qd_all_dofs','hand_world','finger_world')):
                raise ContractError('Image checkpoint arm observation is stale')
        try:
            # This source fixture is inside the west/south cell walls. Keep
            # empty-path inspection cameras inside that cell and below the
            # upper rack shelf. Existing mission views remain unchanged.
            offsets=(((.25,-.38,.32),(.28,-.30,.20)) if checkpoint.startswith('empty_')
                else ((-.68,-.82,.72),(.62,-.75,.63)))
            for index,offset in enumerate(offsets):
                position=body+vector(offset);lookat=body+np.array((0.,0.,.035))
                self.camera.set_pose(pos=position,lookat=lookat,up=(0.,0.,1.))
                filename=f'checkpoint_{checkpoint}_{index+1}.png'
                Image.fromarray(self.camera.render(force_render=True)[0]).save(self.output/filename)
                record={'checkpoint':checkpoint,'file':filename,'native_scene_step':clock[0],'simulation_s':clock[1],
                    'observation_sha256':digest(observation),'detail':copy.deepcopy(detail),
                    'camera':{'pos':position.tolist(),'lookat':lookat.tolist(),'up':[0.,0.,1.]},
                    'image_sha256':hashlib.sha256((self.output/filename).read_bytes()).hexdigest(),
                    'source':'Raw camera.render(force_render=True) RGB, PNG encoding only. No overlay, compositing or replay.'}
                entries.append(record)
        finally:self.camera.set_pose(**saved)
        difference=max(float(np.max(np.abs(vector(getattr(self.camera,'get_'+k)())-value))) for k,value in saved.items())
        if difference>1e-6:raise ContractError('Camera-only restoration differs from its original pose')
        after=self.observe()
        if before!=after or clock!=(int(self.scene.t),float(self.scene.cur_t)):
            raise ContractError('Native physical state changed during image capture')
        for record in entries:record.update(source_state_unchanged_after_capture=True,camera_restored=True,
            camera_restore_max_abs_difference=difference)
        (self.output/(checkpoint+'__frame_observation.json')).write_text(json.dumps(observation,indent=2)+'\n',encoding='utf-8')
        self.records.extend(entries)
        (self.output/'sparse_image_checkpoints.json').write_text(json.dumps(self.records,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return entries
