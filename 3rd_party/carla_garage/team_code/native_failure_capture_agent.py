"""Capture exact student inputs for an isolated, offline reference teacher.

The reference never runs in the driving process and never supplies controls.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from sensor_agent import SensorAgent
from diagnostic_logger import plain, control_values
from learned_teacher_labels import TeacherLabelWriter


def get_entry_point():
    return 'NativeFailureCaptureAgent'


class NativeFailureCaptureAgent(SensorAgent):
    def setup(self,path_to_conf_file,route_index=None,traffic_manager=None):
        self.capture_writer=None
        super().setup(path_to_conf_file,route_index,traffic_manager)
        if len(self.nets)!=1 or self.config.use_wp_gru or self.config.detect_boxes:
            raise ValueError('Exact-input collection requires one native trajectory/speed model')
        route_name=path_to_conf_file.rsplit('+',1)[-1]
        root=Path(os.environ['SHADOW_COLLECTION_ROOT'])
        directory=root/'Recovery_Train'/route_name.replace('_rep0_','_Rep0_')
        self.capture_writer=TeacherLabelWriter(directory,{'phase':'unlabelled_student_states',
            'reference_teacher_checkpoint':os.environ['SHADOW_TEACHER_CHECKPOINT'],
            'student_model':str(next(Path(self.config_path).glob('model*.pth')).resolve()),
            'teacher_control_applied':False,'label_valid_until_offline_annotation':False,
            'benchmark_routes_used_for_development':True,'sample_every_ticks':self.config.data_save_freq},self.config)
        self.capture_inputs=None
        (directory/'exact_inputs').mkdir()
        original=self.nets[0].forward
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            if self.step % self.config.data_save_freq == 0:
                self.capture_inputs={k:v.detach().cpu().clone() for k,v in kwargs.items() if torch.is_tensor(v)}
            return result
        self.nets[0].forward=capture

    def _record_diagnostics(self,input_data,timestamp,tick_data,control,**policy):
        # Additional observations only; normal inference/control remains intact.
        super()._record_diagnostics(input_data,timestamp,tick_data,control,**policy)
        writer=self.capture_writer
        if writer is None or policy.get('phase')!='inference' or self.step % self.config.data_save_freq:
            return
        if self.capture_inputs is None:raise ValueError('Missing exact inputs for sampled inference tick')
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        world=CarlaDataProvider.get_world();ego=CarlaDataProvider.get_hero_actor();transform=ego.get_transform()
        frame=writer.count;path=writer.directory/'exact_inputs'/f'{frame:04d}.pt'
        torch.save(self.capture_inputs,path)
        command=self.commands[-2]
        zero=np.zeros((self.config.fmsr_num_poses,2),dtype=np.float32)
        measurement={'label_source':'learned_teacher','label_valid':False,'teacher_checkpoints':zero,
            'route':zero,'target_speed':0.,'brake':1,'steer':0.,'throttle':0.,'angle':0.,
            'light_hazard':False,'stop_sign_hazard':False,'junction':False,
            'speed':float(tick_data['speed'].item()),'theta':tick_data['compass'],'pos_global':tick_data['gps'],
            'command':command,'next_command':command,'target_point':tick_data['target_point'].cpu().numpy().reshape(2),
            'target_point_next':tick_data.get('target_point_next',tick_data['target_point']).cpu().numpy().reshape(2),
            'augmentation_rotation':0.,'augmentation_translation':0.,'student_step':self.step,
            'student_route':plain(policy['predicted_trajectory_ego_m']),
            'student_target_speed':policy['control_target_speed_mps'],'student_control':control_values(control),
            'ego_world_position_m':[transform.location.x,transform.location.y,transform.location.z],
            'ego_yaw_deg':transform.rotation.yaw,
            'sensor_frames':{name:int(value[0]) for name,value in input_data.items()},
            'exact_inputs_file':str(path.resolve())}
        # Only diagnostics may use simulator pose; neither policy nor teacher sees it.
        lidar=self.lidar_buffer[-1] if self.lidar_buffer else tick_data['lidar']
        writer.write(frame,world.get_snapshot().frame,input_data['rgb_front'][1],lidar,measurement,
            {'reasons':[],'label_valid':False,'student_step':self.step})
        self.capture_inputs=None

    def destroy(self,results=None):
        if self.capture_writer is not None:self.capture_writer.close()
        super().destroy(results)
