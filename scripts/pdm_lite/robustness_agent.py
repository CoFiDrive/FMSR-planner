"""Production FM agent with isolated, audited experiment hooks."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from sensor_agent import SensorAgent
from scripts.pdm_lite.robustness_protocol import Perturbation,attach_head,attach_bridge_planner,route_identity


def get_entry_point():return 'RobustnessAgent'


class RobustnessAgent(SensorAgent):
    def setup(self,path_to_conf_file,route_index=None,traffic_manager=None):
        manifest=json.loads(Path(os.environ['ROBUSTNESS_SPEC']).read_text())
        route=route_identity(path_to_conf_file)
        spec=(manifest['route_conditions'][route.split('_')[1]]
              if 'route_conditions' in manifest else manifest['perturbation'])
        self.perturb=Perturbation(spec,manifest['seed'],route)
        random.seed(manifest['seed']);np.random.seed(manifest['seed']);torch.manual_seed(manifest['seed'])
        torch.cuda.manual_seed_all(manifest['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        self._robust_start=None;self._robust_actor_tick=-1
        super().setup(path_to_conf_file,route_index,traffic_manager)
        for net in self.nets:
            head=getattr(net,'fmsr_head',None)
            if head is None:
                head=net.planner
                if net.training or head.training:raise RuntimeError('Evaluation requires eval mode')
                attach_bridge_planner(head,self.perturb)
                continue
            if net.training or head.training:raise RuntimeError('Evaluation requires eval mode')
            attach_head(head,self.perturb)
        if self.model_count!=1:raise ValueError('Perturbation suite requires a single checkpoint')
        if self.eval_metric_logger is None:raise ValueError('Missing durable metric logger')
        directory=self.eval_metric_logger.directory
        self._robust_audit=(directory/'perturbations.jsonl').open('a',buffering=1)
        self._robust_actors=(directory/'actors.jsonl').open('a',buffering=1)
        (directory/'robustness_spec.json').write_text(json.dumps(manifest,indent=2))

    def _perturb_localization(self,state,compass):
        return self.perturb.localization(state,compass)

    def run_step(self,input_data,timestamp,sensors=None):
        if self._robust_start is None:self._robust_start=float(timestamp)
        self.perturb.begin(float(timestamp)-self._robust_start)
        control=super().run_step(input_data,timestamp,sensors)
        if self.perturb.events:
            self._robust_audit.write(json.dumps(dict(step=self.step,timestamp_s=float(timestamp),
                                      events=self.perturb.events))+'\n')
        if self.step%2==0:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world=CarlaDataProvider.get_world();ego=CarlaDataProvider.get_hero_actor()
            actors=[];origin=ego.get_location()
            for actor in world.get_actors():
                if actor.id==ego.id or not actor.type_id.startswith(('vehicle.','walker.','traffic.')):continue
                location=actor.get_location()
                if location.distance(origin)>50:continue
                transform=actor.get_transform();velocity=actor.get_velocity()
                entry=dict(id=actor.id,type=actor.type_id,position_m=[location.x,location.y,location.z],
                           yaw_deg=transform.rotation.yaw,velocity_mps=[velocity.x,velocity.y,velocity.z])
                if actor.type_id.startswith(('vehicle.','walker.')):
                    box=actor.bounding_box
                    entry['bbox_extent_m']=[box.extent.x,box.extent.y,box.extent.z]
                if hasattr(actor,'get_state'):entry['traffic_state']=str(actor.get_state())
                actors.append(entry)
            self._robust_actors.write(json.dumps(dict(step=self.step,timestamp_s=float(timestamp),actors=actors))+'\n')
        return control

    def _record_diagnostics(self,input_data,timestamp,tick_data,control,**policy):
        # Keep sufficient per-tick behavior data without saving 60-mode arrays or 20-step tensors.
        keys=('available','head','selected_mode','nfe','planner_gpu_ms','planner_wall_submit_ms',
              'candidate_count','generated_mode_count','trajectory_coordinates')
        self._eval_generation=[{k:g[k] for k in keys if k in g} for g in getattr(self,'_eval_generation',[])]
        policy['robustness']=dict(relative_time_s=self.perturb.time,events=self.perturb.events)
        return super()._record_diagnostics(input_data,timestamp,tick_data,control,**policy)

    def destroy(self,results=None):
        try:return super().destroy(results)
        finally:
            for name in ('_robust_audit','_robust_actors'):
                stream=getattr(self,name,None)
                if stream is not None:stream.close()
