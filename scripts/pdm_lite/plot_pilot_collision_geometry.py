"""Offline diagnostic plots; simulator actor data never feeds the driving policy."""
import json
import math
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/fmsr-pilot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from analyze_tfpp_speed_pilot import DEFAULT, ANALYSIS


def box(actor,origin):
    angle=math.radians(actor['rotation_deg'][2]);c,s=math.cos(angle),math.sin(angle);rot=np.array([[c,-s],[s,c]])
    x,y=actor['bbox_extent_m'][:2];corners=np.array([[-x,-y],[x,-y],[x,y],[-x,y]])
    return (corners+np.array(actor['bbox_offset_m'][:2]))@rot.T+np.array(actor['position_m'][:2])-origin


def main():
    cp=json.loads((DEFAULT/'checkpoint_endpoint.json').read_text())['_checkpoint']
    for id in ('1852','2201','2534'):
        route=next((r for r in cp['records'] if r['route_id']==f'RouteScenario_{id}_rep0'),None)
        if route is None:continue
        folder=DEFAULT/'diagnostics'/route['save_name'];events=[json.loads(l) for l in (folder/'events.jsonl').read_text().splitlines()];event=next((e for e in events if 'COLLISION' in e['type']),None)
        if event is None:continue
        rows=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()];f=event['event_frame'];actor_id=event['data']['other_actor']['id'];fig,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,dt in zip(axs,(-40,-20,0)):
            row=min(rows,key=lambda r:abs(r['simulation_frame']-(f+dt)));ego=row['ego'];origin=np.array(ego['position_m'][:2]);angle=math.radians(ego['rotation_deg'][2]);c,s=math.cos(angle),math.sin(angle);rot=np.array([[c,-s],[s,c]])
            ax.add_patch(Polygon(box(ego,origin),facecolor='#258c75',alpha=.7))
            for actor in row['nearby_actors']:
                if not actor['type'].startswith(('vehicle.','walker.')):continue
                ax.add_patch(Polygon(box(actor,origin),facecolor='#d75252' if actor['id']==actor_id else '#aaaaaa',alpha=.6))
            path=np.array(row['policy']['predicted_trajectory_ego_m'])@rot.T;ax.plot(path[:,0],path[:,1],'o-',ms=3,color='#1565c0',label='FM path')
            speed=row['policy']['speed_comparison'][0];ax.set_title(f"{dt*.05:+.1f} s; actual {row['policy']['measured_speed_mps']:.1f}\nFM {speed['planner_speed_mps'][0]:.1f}, TF++ {speed['tfpp_speed_mps'][0]:.1f} m/s")
            ax.set_xlim(-20,20);ax.set_ylim(-20,20);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('World x relative to ego (m)')
        axs[0].set_ylabel('World y relative to ego (m)');fig.suptitle(f"{route['scenario_name']} / {id}: green ego, red collision actor; recorded visited states");fig.tight_layout();fig.savefig(ANALYSIS/f'collision_{id}_geometry.png',dpi=160);plt.close(fig)
    print('Saved collision geometry plots')


if __name__=='__main__':main()
