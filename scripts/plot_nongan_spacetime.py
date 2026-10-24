"""Measured x-y-time trajectories and width ribbons; no invented planning times."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/nongan_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def read(path):
    rows=[]
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows


def draw(directory,traffic,plans,start,duration,stem):
    rows=[r for r in traffic if start<=r['sim_time']<=start+duration]
    origin=next(r['ego'] for r in traffic if r.get('ego'))
    x0,y0=origin['x'],origin['y']
    yaw0=origin['theta']
    rotation=np.array([[math.cos(yaw0),math.sin(yaw0)],[-math.sin(yaw0),math.cos(yaw0)]])
    def to_reference(points):return (np.asarray(points)-[x0,y0])@rotation.T
    trajectories={}
    for row in rows:
        objects=([row['ego']] if row.get('ego') else [])+row.get('actors',[])
        for actor in objects:
            identity=actor['actor_id']
            trajectories.setdefault(identity,[]).append((row['sim_time']-start,row['frame'],actor))
    if directory.parent.name=='left_combined':
        lo=np.array([0.,-25.]);hi=np.array([150.,120.])
    else:
        all_positions=to_reference([[actor['x'],actor['y']]
            for samples in trajectories.values() for _,_,actor in samples])
        lo=all_positions.min(axis=0)-5.
        hi=all_positions.max(axis=0)+5.
    local_style=stem=='01_spacetime_lane_changes'
    fig=plt.figure(figsize=(1100/180,900/180) if local_style else (9,8))
    axes_position=([-.02,.11,.94,.90] if local_style and directory.parent.name=='roundabout'
                   else [.02,.11,.96,.90] if local_style else [.03,.03,.94,.94])
    ax=fig.add_axes(axes_position,projection='3d')
    palette=['#007f5f','#e69f00','#8e44ad','#0096c7','#a98467','#7f7f00']
    npc_index=0;handles=[]
    crop=True
    def in_bounds(xy):
        xy=np.asarray(xy)
        return (xy[...,0]>=lo[0]-1e-3)&(xy[...,0]<=hi[0]+1e-3)&(xy[...,1]>=lo[1]-1e-3)&(xy[...,1]<=hi[1]+1e-3)
    for identity,samples in trajectories.items():
        ego=samples[0][2]['role_name']=='ego_vehicle'
        color='#cf3030' if ego else palette[npc_index%len(palette)]
        if not ego:npc_index+=1
        role=samples[0][2]['role_name'].split('/')[-1]
        labels={'ego_vehicle':'Ego vehicle','lane1_car':'Lane 1 vehicle',
                'lane2_car':'Lane 2 vehicle','west1':'West 1','west2':'West 2',
                'east1':'East 1','east2':'East 2',
                'inner1':'Inner 1','inner2':'Inner 2',
                'outer1':'Outer 1','outer2':'Outer 2',
                'outer3':'Outer 3','outer4':'Outer 4'}
        handles.append(Line2D([0],[0],color=color,lw=2.4 if ego else 1.6,
                              label=labels.get(role,role.replace('_',' ').title())))
        positions=to_reference([[a['x'],a['y']] for _,_,a in samples])
        points=np.column_stack([positions,[t for t,_,_ in samples]])
        visible=points.copy()
        if crop:visible[~in_bounds(positions)]=np.nan
        ax.plot(*visible.T,color=color,lw=2.4 if ego else 1.6)
        ax.plot(visible[:,0],visible[:,1],np.zeros(len(visible)),color=color,lw=.9,alpha=.35)
        if np.isfinite(visible[0]).all():ax.scatter(*visible[0],color=color,s=22,marker='^')
        if np.isfinite(visible[-1]).all():ax.scatter(*visible[-1],color=color,s=22,marker='o')
        # Ribbon uses true half width, without enlarging it for presentation.
        # Do not bridge missing simulation frames.
        faces=[]
        for (t,frame,a),(nt,nframe,b) in zip(samples[:-1],samples[1:]):
            if nframe!=frame+1:continue
            def edges(actor,time):
                center=actor['bbox']['center'];yaw=actor['bbox']['theta']
                width=actor['bbox']['extent'][1]
                dx,dy=-math.sin(yaw)*width,math.cos(yaw)*width
                positions=to_reference([[center[0]+dx,center[1]+dy],[center[0]-dx,center[1]-dy]])
                return [[*positions[0],time],[*positions[1],time]]
            left,right=edges(a,t);nleft,nright=edges(b,nt)
            face=[left,right,nright,nleft]
            if not crop or in_bounds(np.asarray(face)[:,:2]).all():faces.append(face)
        if faces:ax.add_collection3d(Poly3DCollection(faces,facecolors=color,alpha=.16 if ego else .12,edgecolors='none'))
    # Three spatial plans at their actual publication time, not predicted future time.
    selected=([p for p in plans if p.get('published_path_world_m') and
              start<=p['sim_time']+p['output_age_s']<=start+duration]
              if not local_style else [])
    if selected:
        indices=sorted(set(int((len(selected)-1)*q) for q in (.2,.5,.8)))
        plan_color='#245bc2'
        for index in indices:
            p=selected[index];path=to_reference(p['published_path_world_m'])
            stamp=p['sim_time']+p['output_age_s']-start
            if crop:path[~in_bounds(path)]=np.nan
            ax.plot(path[:,0],path[:,1],np.full(len(path),stamp),color=plan_color,ls='--',lw=1.3)
        handles.append(Line2D([0],[0],color=plan_color,ls='--',lw=1.3,label='Local plan'))
    top=max(row['sim_time']-start for row in rows)
    plane=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]
    for height,alpha in [(0.,.08),(top,.035)]:
        ax.add_collection3d(Poly3DCollection([[[x,y,height] for x,y in plane]],
            facecolors='gray',alpha=alpha,edgecolors='gray',linewidths=.5))
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(0.,top*1.02),
        xlabel='x (m)',ylabel='y (m)',zlabel='t (s)')
    if directory.parent.name=='left_combined':
        ax.set_xticks([0,25,50,75,100,125,150])
    if directory.parent.name=='left_combined':ax.set_yticks([-25,0,25,50,75,100,120])
    else:ax.set_yticks([-30,30])
    if local_style:ax.tick_params(labelsize=11)
    ax.xaxis.set_tick_params(pad=2)
    ax.yaxis.set_tick_params(pad=4 if local_style and directory.parent.name=='roundabout' else 14)
    ax.yaxis.labelpad=35 if directory.parent.name=='left_combined' else 25
    if local_style and directory.parent.name=='roundabout':
        ax.zaxis.labelpad=2
        ax.set_ylabel('')
        fig.text(.90,.075,'y (m)',rotation=38,ha='center',va='center',fontsize=14)
    if directory.parent.name=='left_combined':
        if local_style:ax.set_box_aspect((1.2,1.2,1.0),zoom=1.10)
        else:ax.set_box_aspect((1.2,1.2,1.25))
    else:
        span=hi-lo
        ax.set_box_aspect((span[0],span[1],max(span[0]*.5,top)),
                          zoom=1.17 if local_style and directory.parent.name=='roundabout' else 1.02)
    ax.view_init(elev=24,azim=-55)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis._axinfo['grid']['color']=(.75,.75,.75,.25)
    ax.xaxis.pane.fill=False;ax.yaxis.pane.fill=False;ax.zaxis.pane.fill=False
    legend_style=dict(handles=handles,loc='upper left',ncol=2,
                      fontsize=11 if local_style else 13,framealpha=1.0,
                      columnspacing=.55 if local_style else 1.0,
                      handlelength=1.25 if local_style else 1.8,
                      handletextpad=.35 if local_style else .8,
                      labelspacing=.15 if local_style else .3,
                      borderpad=.2 if local_style else .35)
    if stem=='01_spacetime_full':
        fig.legend(bbox_to_anchor=(.265,.71),bbox_transform=fig.transFigure,**legend_style)
    else:
        if local_style and directory.parent.name=='roundabout':
            legend_style['loc']='lower left'
            position=(.04,.06)
        else:
            position=(.20,.74) if directory.parent.name=='left_combined' else (.03,.93)
        fig.legend(bbox_to_anchor=position,bbox_transform=fig.transFigure,**legend_style)
    output=directory/'analysis_figures';output.mkdir(exist_ok=True)
    for suffix in ('png','svg'):
        if local_style:fig.savefig(output/(stem+'.'+suffix),dpi=180)
        else:fig.savefig(output/(stem+'.'+suffix),dpi=180,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path);args=parser.parse_args()
    directory=args.run_dir.resolve()
    plt.rcParams['font.family']='Arial'
    plt.rcParams['font.size']=14
    plt.rcParams['axes.labelsize']=14
    plt.rcParams['xtick.labelsize']=14
    plt.rcParams['ytick.labelsize']=14
    plt.rcParams['svg.fonttype']='none'
    plt.rcParams['axes.unicode_minus']=False
    traffic=read(directory/'traffic_state.jsonl');plans=read(directory/'planning.jsonl')
    start=next(r['sim_time'] for r in traffic if r.get('ego'))
    duration=traffic[-1]['sim_time']-start
    draw(directory,traffic,plans,start,duration,'01_spacetime_full')
    last_actor_time=max((r['sim_time'] for r in traffic if r.get('actors')),default=start)
    lane_changes_duration=min(duration,max(20.,last_actor_time-start+2.))
    draw(directory,traffic,plans,start,lane_changes_duration,'01_spacetime_lane_changes')
    (directory/'analysis_figures/spacetime_notes.json').write_text(json.dumps(dict(
        run=str(directory),time_origin_sim_time=start,
        representation='Measured center trajectories with physical width ribbons; vehicle length not represented',
        coordinates='Fixed reference: origin at first ego position, x along initial ego heading, y left; never a moving frame',
        planning_time='Only the full plot shows spatial plans; their points share each plan publication timestamp, with no inferred future point timestamps',
        lane_changes_duration_s=lane_changes_duration,
        lane_changes_window='From first ego observation through last observed traffic actor plus 2 seconds, at least 20 seconds and no longer than the run',
        files=['01_spacetime_full.png','01_spacetime_lane_changes.png']),indent=2)+'\n')
    print(directory/'analysis_figures/01_spacetime_full.png')
    print(directory/'analysis_figures/01_spacetime_lane_changes.png')


if __name__=='__main__':main()
