"""Summarize same-state reference evidence and visualize failed-turn windows."""
import os,json,math
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/fmsr-shadow-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
A=ROOT/'outputs/analysis/fm_failure_state_teacher_20261008'
DATA=ROOT/'outputs/local_recovery/fm_failure_state_teacher_20261008'


def main():
    labels=json.loads((A/'label_verification.json').read_text());repeat=json.loads((A/'teacher_repeatability.json').read_text())
    representation=json.loads((A/'route_representation_audit.json').read_text());rows=[json.loads(s) for s in (A/'same_state_reference.jsonl').read_text().splitlines()]
    near_alerts=[r for r in representation['frames'] if abs(r['matched_heading_gap_deg'])>15. and r['matched_near3_ADE_m']>.75]
    (A/'near_heading_review_frames.json').write_text(json.dumps({'thresholds':{'heading_gap_deg':15,'near3_ADE_m':.75},'scope':'review selection only, not validation score or training eligibility','frames':near_alerts},indent=2)+'\n')
    all_pre=[]
    for route in sorted((DATA/'Recovery_Train').iterdir()):
        meta=json.loads((route/'collection_metadata.json').read_text());meta.update(phase='reference_labelled_pending_quality_review',training_eligible=False)
        (route/'collection_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        events=[json.loads(s) for s in (route/'recovery_events.jsonl').read_text().splitlines()]
        all_pre.extend(f'Recovery_Train/{route.name}/{e["frame"]:04d}' for e in events if e['pre_collision'])
    (A/'pre_collision_review_frames.txt').write_text('\n'.join(all_pre)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for ax,case in zip(axs,representation['cases']):
        id=case['route_id'];example=case['example'];r=next(r for r in rows if r['route_id']==id and r['frame']==example['frame'])
        student=np.asarray(r['student_path']);teacher=np.asarray(r['teacher_path']);matched=np.asarray(example['teacher_matched_path'])
        ax.plot(*student.T,'o-',label='Student FM path',color='#d98c35',ms=3);ax.plot(*teacher.T,'o-',label='Clean-prior reference',color='#1565c0',ms=3)
        ax.plot(*matched.T,':',label='Legacy FM label sampling',color='#888888')
        ax.plot(*np.asarray(example['teacher_ordered_path']).T,'--',label='Ordered polyline sampling',color='#258c75')
        ax.scatter(0,0,color='black',s=50,label='Ego');ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('Ego forward x (m)');ax.set_ylabel('Ego right y (m)')
        ax.set_title(f"{id}; frame {r['frame']}\nordered lookahead heading gap {example['ordered_heading_gap_deg']:.1f} deg");ax.legend(fontsize=8)
    fig.suptitle('Pre-collision visited states; reference is a learned prediction, not ground truth');fig.tight_layout();fig.savefig(A/'same_state_turn_paths.png',dpi=160);plt.close(fig)
    text=['# 失败状态采集与同输入参考教师测试（2026-10-08）','',
      '完成2条学生闭环复现；学生保持FM路径+TF++原速度，教师不参与车辆控制。保存实际网络输入，采集结束后在独立进程对同输入运行clean-prior epoch09完整20步生成。', '',
      '| 路线 | 本次学生DS | 状态 | 保存帧 | 首次碰撞前帧 |','| --- | ---: | --- | ---: | ---: |']
    for c in labels['cases']:text.append(f"| {c['route_id']} | {c['student_DS']:.4f} | {c['student_status']} | {c['saved_frames']} | {c['pre_collision_frames']} |")
    text += ['',f"共保存 {sum(c['saved_frames'] for c in labels['cases'])} 帧；首次碰撞前 {len(all_pre)} 帧。原分歧规则标记14帧，新增近端航向审查规则标记 {len(near_alerts)} 帧。全部预碰撞帧保留，不仅保留触发帧。",'',
      '## 新证据','',
      '1. **近端转向路径分歧。** 原始教师点序和FM采样表示不同，直接点对点ADE不能全归为路径质量。把教师路径转换为当前FM标签约定（首点距ego2.5m、后续1m间距）后，有效lookahead航向分歧仍明显：', '',
      '| 路线 | 全部预碰撞匹配表示ADE | 碰撞前8帧平均近3点分歧 | 碰撞前8帧最大航向分歧 |','| --- | ---: | ---: | ---: |']
    for c in representation['cases']:text.append(f"| {c['route_id']} | {c['mean_matched_ADE_m']:.4f} m | {c['last8_mean_matched_near3_ADE_m']:.4f} m | {c['last8_max_abs_heading_gap_deg']:.2f}° |")
    text += ['',
      '23901示例：学生有效航向约-13.19°，匹配教师约-54.66°；24041示例为-5.21°对-50.18°。参考路径更早朝内侧转向，学生近端点仍偏向直行。方向差异不是单纯首点距离不同造成；目前尚未证明教师接管这些异常状态能够安全恢复。', '',
      '当前route_geometry.iterative_line_interpolation会沿当前射线延伸到2.5m，再逐点做圆线交点；对短首点的急弯参考路径也会产生局部回折。因此统一表示结果用于排除采样差异，不能将这个重采样结果直接作为安全专家标签。原保存教师点未重采样，生产loader也验证了逐点保留。', '',
      '新增独立可选函数sample_route_on_polyline：沿原折线按顺序寻找距离圆交点，只在路线耗尽时沿末端切线外推。3项测试验证直线/外推、急弯不折返且采样点位于原折线、无效输入拒绝。旧smooth_route未替换，已有anchors和checkpoint语义保持。', '',
      '有序采样之后的分歧仍存在：23901最后8个预碰撞状态平均近3点分歧1.0655m、最大有效航向分歧45.53°；24041分别1.6002m和47.62°。这支持优先检查近端路径和lookahead配合，不能把分歧全部解释为旧采样的局部折返。该新函数尚未证明闭环收益；更换旧训练标签需同步重建相关assets并重新验证。', '',
      '2. **教师速度标签不稳定，不能直接复制。** 对每条路线最后8个预碰撞状态，用随机种子17和41完整生成：', '',
      '| 路线 | 最大路径坐标差 | 最大速度差 |','| --- | ---: | ---: |']
    for r in repeat:text.append(f"| {r['route_id']} | {r['max_path_coordinate_difference_m']:.6f} m | {r['max_speed_difference_mps']:.4f} m/s |")
    text += ['',
      '路径差小于毫米，速度差分别约8.62和0.82m/s。clean-prior在自己的闭环状态上曾成功，不能据此假设它在学生异常状态上的速度标签也可靠。重复性只是质量检查，路径重复并不证明可通行。', '',
      '3. **事件选择会漏掉近端转向差异。** 现有plan_disagreement仅看前6点平均距离>2.5m，23901的一帧虽有约41°有效航向差，也未触发这个条件。新增15°且近3点>0.75m的规则仅用于审查窗口，不作为改进指标、训练门槛或部署规则。', '',
      '## 后续方向','',
      '- 优先验证路径修正：在预碰撞急弯状态检查近端航向、首点距离和控制器lookahead；用实际专家或受控接管验证参考路径能否恢复，然后做小规模路径监督。',
      '- 速度标签需单独评估生成方差/停车语义；保留冻结TF++速度候选，不把当前参考模型的单次随机速度直接蒸馏到新头。',
      '- 新速度头应接专用速度query、真实生成路径和感知障碍特征；现已保存RGB/LiDAR/navigation精确输入，可以重新计算特征，无需用诊断中的GT actor数据作为推理输入。',
      '- 时序验证用实际学生状态序列，检查路径随ego移动后的几何一致性与速度跳变；本次随机种子对照仅验证同状态生成重复性，尚未替代跨帧时序或教师接管闭环。', '',
      '## 数据与完整性','',
      '所有639帧RGB/LiDAR文件完整解码、传感器帧号对齐。6个真实样本经生产CARLA_Data读回，教师路径和速度逐项保留，输入为RGB[3,384,1024]、LiDAR[1,256,256]。4项捕获/标签读回/训练质量排除测试及3项有序采样测试通过。', '',
      '未进行新训练；training_eligible=False使本批参考标签不进入通用恢复训练manifest。155个预碰撞帧及所有后碰撞帧均保留作诊断，不能把后碰撞长时间停滞当作大量独立纠正样本。验证专用manifest只用于读回检查。', '',
      '这两条是已知benchmark失败路线，属于开发诊断数据；不构成独立holdout。没有教师控制接管，学生参数与默认策略保持。', '',
      '## 文件','',
      '- protocol.json、label_verification.json、native_loader_verification.json：协议与真实数据检查。',
      '- same_state_reference.jsonl、route_representation_audit.json、teacher_repeatability.json：每帧配对和生成重复性。',
      '- pre_collision_review_frames.txt：155个预碰撞帧审查清单；near_heading_review_frames.json：近端航向窗口。',
      '- same_state_turn_paths.png：近端路径对照。',
      '- ../../local_recovery/fm_failure_state_teacher_20261008：原始观测、exact_inputs及参考标签。',
      '- ../../local_evaluation/bench2drive2_fm_tfpp_failure_capture_20261008：完成的学生闭环评分和diagnostics。']
    (A/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'saved_frames':sum(c['saved_frames'] for c in labels['cases']),'pre_collision_frames':len(all_pre),'near_heading_review_frames':len(near_alerts),'report':str(A/'report.md')},indent=2))


if __name__=='__main__':main()
