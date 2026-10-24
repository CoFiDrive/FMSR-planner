"""Plot recorded driving trajectories and FM paths before terrain collisions."""
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fmsr-pilot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from analyze_tfpp_speed_pilot import DEFAULT
from analyze_curvature_pilot import OUT, REPORT


def main():
    for id in ('23901','24041'):
        fig,axs=plt.subplots(1,2,figsize=(10,5));found=False
        for ax,root,label in zip(axs,(DEFAULT,OUT),('TF++ speed','TF++ + curvature cap')):
            cp=json.loads((root/'checkpoint_endpoint.json').read_text())['_checkpoint'];r=next((r for r in cp['records'] if r['route_id']==f'RouteScenario_{id}_rep0'),None)
            if r is None:continue
            folder=root/'diagnostics'/r['save_name'];es=[json.loads(l) for l in (folder/'events.jsonl').read_text().splitlines()];event=next((e for e in es if e['type']=='COLLISION_STATIC'),None)
            if event is None:continue
            found=True;f=event['event_frame'];qs=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()];prior=[q for q in qs if f-200<=q['simulation_frame']<=f];xy=np.array([q['ego']['position_m'][:2] for q in prior]);origin=np.array([-2580.,5910.]);ax.plot(*(xy-origin).T,label='Actual trajectory',color='#258c75');ax.scatter(*(np.array(event['data']['location'][:2])-origin),color='#d75252',marker='x',s=80,label='Terrain collision')
            for offset in (-40,-20):
                q=min(qs,key=lambda q:abs(q['simulation_frame']-(f+offset)));yaw=math.radians(q['ego']['rotation_deg'][2]);c,s=math.cos(yaw),math.sin(yaw);rot=np.array([[c,-s],[s,c]]);path=np.array(q['policy']['predicted_trajectory_ego_m'])@rot.T+np.array(q['ego']['position_m'][:2])-origin;ax.plot(*path.T,'--',label=f'FM prediction {offset*.05:.0f}s before',alpha=.8)
            at=min(qs,key=lambda q:abs(q['simulation_frame']-f));ax.set_title(f"{label}\nCollision speed {abs(at['policy']['measured_speed_mps']):.2f} m/s");ax.set_xlabel('World x +2580 (m)');ax.set_ylabel('World y -5910 (m)');ax.set_xlim(-25,35);ax.set_ylim(-35,25);ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(fontsize=8)
        if found:fig.suptitle(f'Route {id}: recorded trajectories; road boundary is not shown');fig.tight_layout();fig.savefig(REPORT/f'terrain_{id}_replays.png',dpi=160)
        plt.close(fig)
    print('Saved available terrain replay plots')


if __name__=='__main__':main()
