"""Four-column, event-anchored qualitative cases with explicit run provenance."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, findfont
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from render_visual_cases import records, project_path

BASE = Path('/media/vcdc/T71/fmsr_visual_cases_20261021')
MODEL_NAMES = ['FMSR', 'FM', 'FM_CR']
LABELS = {'FMSR': 'FMSR', 'FM': 'FM', 'FM_CR': 'FM+SR'}
COLORS = {'FMSR': '#F04438', 'FM': '#2454E8', 'FM_CR': '#39CC24'}
ROUTES = ['28093', '17563', '2204', '24785']


def source(model, rid):
    if rid == '2204' and model == 'FM':
        return Path('/media/vcdc/T71/fmsr_visual_cases_2204_retest_20261021')
    if rid == '2204' and model == 'FM_CR':
        return Path('/media/vcdc/T71/fmsr_visual_cases_2204_retest02_20261021')
    return BASE


def load_case(model, rid):
    root = source(model, rid)
    directory = next((root / model / 'diagnostics').glob('RouteScenario_' + rid + '_rep0*'))
    frames = records(directory / 'frames.jsonl')
    indexed = {r['simulation_frame']: r for r in frames}
    visual = [r for r in records(directory / 'visual/frames.jsonl')
              if r['simulation_frame'] in indexed]
    events = [r for r in records(directory / 'events.jsonl') if r['type'].startswith('COLLISION')]
    result = next(r for r in json.loads((root/model/'checkpoint_endpoint.json').read_text())['_checkpoint']['records']
                  if r['route_id'] == 'RouteScenario_' + rid + '_rep0')
    spec = next(s for s in json.loads((directory/'visual/camera_specs.json').read_text()) if s['id']=='visual_third')
    return dict(root=root, directory=directory, frames=frames, indexed=indexed,
                visual=visual, events=events, result=result, spec=spec)


def camera(ax, case, selected, color=None):
    ax.imshow(Image.open(case['directory']/'visual'/selected['images']['third']))
    if color:
        row = case['indexed'][selected['simulation_frame']]
        points = project_path(row['policy'], selected['ego_matrix'],
                              selected['camera_world_matrices']['third'], case['spec'])
        if len(points):
            ax.plot(points[:,0], points[:,1], '-o', color=color, lw=1.6, ms=2.2)
    ax.set_xlim(0,case['spec']['width'])
    ax.set_ylim(case['spec']['height'],0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('/home/vcdc/fmsr/outputs/analysis/four_visual_cases_paper_20261021'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ['Arial.ttf','Arial_Bold.ttf']:
        fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/'+name)
    findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'font.weight':'bold',
                         'axes.labelweight':'bold','axes.titleweight':'bold',
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    # Match the reference's narrow gutters, square maps, and 16:9 camera panels.
    fig,axes=plt.subplots(5,4,figsize=(12,9.5),gridspec_kw={'height_ratios':[.5625,1,.5625,.5625,.5625]})
    fig.subplots_adjust(left=.035,right=.998,top=.975,bottom=.005,wspace=.025,hspace=.025)
    manifest={'selection':'Selected qualitative examples; not an aggregate performance estimate.',
              '2204_note':'Different reruns: FM first retest; FM+CR second retest; FMSR original. '
                          'FM original DS=100; FM+CR original DS=35.73 and first retest DS=80 are retained.',
              'cases':{}}
    for col,rid in enumerate(ROUTES):
        cases={m:load_case(m,rid) for m in MODEL_NAMES}
        collision_locations=[e['data']['location'][:2] for c in cases.values() for e in c['events']]
        # Route2204 focuses on the blockage area; other cases use recorded contact locations.
        anchor=np.asarray(collision_locations[0] if rid=='2204' else np.mean(collision_locations,axis=0))
        for model,c in cases.items():
            if c['events']:
                event=c['events'][0]
                selected=min(c['visual'],key=lambda r:abs(r['simulation_frame']-event['event_frame']))
                reason='nearest collision event'
            elif c['result']['status'].startswith('Failed'):
                selected=c['visual'][-1]
                reason='last recorded frame before tick limit'
            else:
                selected=min(c['visual'],key=lambda r:np.linalg.norm(np.asarray(r['ego_matrix'])[:2,3]-anchor))
                reason='nearest spatial position to conflict area'
            c['selected']=selected
            c['reason']=reason
        reference=cases['FM_CR']
        context=min(reference['visual'],key=lambda r:abs(r['simulation_frame']-(reference['selected']['simulation_frame']-40)))
        camera(axes[0,col],reference,context)
        axes[0,col].set_title('Scenario ({})'.format(chr(97+col)),fontsize=12,pad=4)
        ax=axes[1,col]
        paths={m:np.asarray([r['ego']['position_m'][:2] for r in c['frames']]) for m,c in cases.items()}
        all_xy=np.concatenate(list(paths.values()))
        center=(all_xy.min(0)+all_xy.max(0))/2
        extent=max(np.ptp(all_xy[:,0]),np.ptp(all_xy[:,1]))+12
        low,high=center-extent/2,center+extent/2
        town=reference['selected']['town']
        lanes=json.loads((BASE/'maps'/(town+'_lanes.json')).read_text())['lanes']
        ax.set_facecolor('#EEEEEE')
        for lane in lanes.values():
            centers=np.asarray([r['center'][:2] for r in lane])
            if len(centers)<2 or np.any(centers.max(0)<low) or np.any(centers.min(0)>high):continue
            left=np.asarray([r['left'] for r in lane]);right=np.asarray([r['right'] for r in lane])
            ax.add_patch(Polygon(np.r_[left,right[::-1]],facecolor='#444444',edgecolor='none'))
            for boundary in [left,right]:ax.plot(boundary[:,0],boundary[:,1],color='white',lw=.45)
        for model in MODEL_NAMES:
            xy=paths[model]
            ax.plot(xy[:,0],xy[:,1],lw=1.8,color=COLORS[model],label=LABELS[model])
            position=np.asarray(cases[model]['selected']['ego_matrix'])[:2,3]
            ax.scatter(*position,marker='^',color=COLORS[model],s=35,edgecolor='black',linewidth=.6,zorder=8)
        ax.set(xlim=(low[0],high[0]),ylim=(high[1],low[1]),aspect='equal')
        legend=ax.legend(loc='upper right',fontsize=8,framealpha=1,fancybox=False,
                         edgecolor='black',handlelength=1.5,borderpad=.5)
        legend.get_frame().set_linewidth(.6)
        manifest['cases'][rid]={}
        for row,model in enumerate(MODEL_NAMES,2):
            c=cases[model];ax=axes[row,col]
            camera(ax,c,c['selected'],COLORS[model])
            result=c['result'];ds=result['scores']['score_composed']
            outcome='Collision' if c['events'] else ('Tick limit' if result['status'].startswith('Failed') else 'No collision')
            # A small badge keeps the mixed rerun outcomes visible in the figure itself.
            ax.text(.02,.96,'{} | DS {:g}'.format(outcome,ds),transform=ax.transAxes,
                    color='white',fontsize=7.5,va='top',bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':2})
            selected=c['selected']
            manifest['cases'][rid][model]={'directory':str(c['directory']),
                'selected_frame':selected['simulation_frame'],'selection_reason':c['reason'],
                'event':c['events'][0] if c['events'] else None,
                'status':result['status'],'scores':result['scores']}
        for ax in axes[:,col]:
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
    for row,label in enumerate(['','HD Maps with Trajectories','FMSR','Flow Matching','+Spectrally Regularized']):
        axes[row,0].set_ylabel(label,fontsize=11,labelpad=4)
    for ext in ['png','pdf','svg']:
        fig.savefig(args.output/('four_scenarios_event_keyframes.'+ext),dpi=600,
                    facecolor='white',bbox_inches='tight',pad_inches=.015)
    plt.close(fig)
    (args.output/'selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'caption.txt').write_text(
        'Selected qualitative cases: (a) unsignalized left-turn merge, route 28093; '
        '(b) sequential lane change, route 17563; (c) blocked intersection, route 2204; '
        '(d) construction obstacle, route 24785. Top: FM+CR scene context; second row: '
        'actual paths on sampled CARLA lane geometry, triangles mark selected ego positions. '
        'Remaining rows: event/spatially selected third-person views and predicted trajectories. '
        'Collision frames are nearest recorded camera frames (0.25s sampling). Runs are independent. '
        'Route 2204 uses different reruns documented in selection_manifest.json, including FM tick-limit termination; '
        'its original FM run scored 100. This selection does not establish consistent superiority.\n')
    print(args.output/'four_scenarios_event_keyframes.png')


if __name__=='__main__':main()
