"""Assemble completed experiments and evidence-based next steps."""
import json
from pathlib import Path
from analyze_tfpp_speed_pilot import ANALYSIS, ROOT
from analyze_curvature_pilot import REPORT


def main():
    a=json.loads((ANALYSIS/'pilot_summary.json').read_text());b=json.loads((REPORT/'summary.json').read_text())
    if not a['complete'] or not b['complete']:
        raise RuntimeError('Both closed-loop experiments must finish before final reporting')
    dst=ROOT/'outputs/analysis/automated_fm_diagnosis_20261008';dst.mkdir(exist_ok=True)
    lines=['# 自动化闭环测试与问题定位（2026-10-08）','',
      '完成两轮独立闭环测试：32条 FM路径+冻结TF++原速度，以及5条固定路线的曲率限速对照。全部原模型权重不变，无新训练，默认agent速度来源仍为planner。', '',
      '## 32条同路线历史对照','', '| 策略 | DS | SR | RC | 成功数 | TickRuntime | 堵塞 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,k in [('原 FM 路径+原 FM速度','baseline_same_routes'),('FM路径+TF++原速度','hybrid')]:
        m=a[k];lines.append(f"| {name} | {m['DS']:.4f} | {m['SR_percent']:.2f}% | {m['RC']:.4f}% | {m['success']}/32 | {m['statuses'].get('Failed - TickRuntime',0)} | {m['statuses'].get('Failed - Agent got blocked',0)} |")
    lines += ['',f"DS增加 {a['hybrid']['DS']-a['baseline_same_routes']['DS']:.4f}，SR增加 {a['hybrid']['SR_percent']-a['baseline_same_routes']['SR_percent']:.2f}个百分点；救回8条、损失4条。",'',
      '分组：12条原失败难路线救回8条，DS45.3128→87.2200；20条原成功普通路线降为16条成功，DS100→93。因此存在明确开发收益和普通路线退化，不能据此直接设为默认。', '',
      '这32条是从已有失败和成功中固定选取的开发集，基线为同路线历史运行。clean-prior在这组全部100分属于选组条件。结果不能外推完整220条，也不是独立holdout或多seed显著性验证。', '',
      '## 5条曲率限速对照','',
      '固定侧向加速度先验3m/s²，从实际FM预测路径取最大三点曲率，速度=min(TF++原速度,sqrt(3/曲率))；其他部分保持。未扫描阈值。', '',
      '| 路线 | TF++速度DS | 加曲率限速DS | 限速结果 |','| --- | ---: | ---: | --- |']
    for q in b['pairs']:lines.append(f"| {q['route_id']} / {q['scenario']} | {q['tfpp_DS']:.4f} | {q['curvature_DS']:.4f} | {q['status']} |")
    lines += ['',f"同5条：DS {b['tfpp_same_routes']['DS']:.4f}→{b['curvature']['DS']:.4f}，成功数 {b['tfpp_same_routes']['success']}/5→{b['curvature']['success']}/5；本版规则未增加成功数，不作为默认、不扩到220条。",'',
      '离线128专家开发帧中25帧被限速，速度MAE0.15573→1.53177m/s。限速可明显改变控制和访问状态，并不保证减少交互碰撞。', '',
      '## 问题在哪里','',
      '1. **速度自由生成存在问题。** 同128专家帧原FM实际速度MAE约0.47217，冻结TF++为0.15573；oracle模式仍约0.4512。GT桥插值误差小不能代表真实rollout。独立零初始化速度修正头500步只改善0.58%，速度行短训练也只有约1%收益。闭环32条中原4条TickRuntime消失，支持原速度分支确实是部分失败因素。',
      '2. **路径/跟踪问题独立存在。** 两条剩余堵塞23901、24041先撞static.terrain，再越线和堵塞。曲率限速后两条依旧失败；碰撞实际速度分别从10.09、13.54m/s降至3.74、2.08m/s。23901停滞时两分支仍要求行驶，原TF++对照存在约60秒目标≥2而实际|速度|<0.5的连续区间。这些不是单纯错误停车。低速仍碰撞支持检查路径几何、近端路径跟踪和导航条件；现有证据没有唯一分离路径预测与控制器责任。',
      '3. **交互时序和速度/路径配合有风险。** 1852、2201原成功路线换速度后碰撞。1852碰撞前16帧TF++≥2而同状态FM<0.5，随后目标约13m/s。2201启动也有显著分歧，但碰撞瞬间两者已接近。2534则在两分支共同要求停车、急减速转向时与邻车接触。不能用单一专家帧速度MAE解释这些交互，也不能从历史单次差异推定唯一因果。',
      '4. **停止标志需要单独审计。** 2416新组合80分、3666为80分，都有stop违规。停止标志控制器启用，但2416违规前未记录stop_sign_brake。需要记录检测框、缓存状态、清除计时及制动触发，区分漏检、提前清除与停车执行不完整；当前证据不能直接定为某一个代码错误。',
      '5. **指标名称有陷阱。** min_speed_infractions按检查点无条件记录周围车速比，再按行驶公里数汇总数量，当前DS罚分类型unused；179.56不能解释成179次错误停车。TickRuntime在本评估为超过4000仿真tick，保留失败计分。诊断速度是有符号纵向速度，停车统计使用绝对值，倒退不是停车。', '',
      '## 下一步改进优先级','',
      '- **先补实际失败状态的数据。** 收集这两条地形路线的学生访问状态及专家纠正路径，核对command/target point、近端路径偏差和控制器lookahead。验证感知BEV上的可通行区域/障碍净空，监督完整rollout的路径；仅增加原专家帧训练或提高路径CR权重没有现有收益证据。',
      '- **把速度和路径/交互联合验证。** 新小头优先使用原TF++专用速度query/logits，加实际FM路径、预测障碍物和时序状态；对启动、合流、避障分歧单独学习。正加速度约束保留紧急减速，不关闭安全覆盖。单纯min会保留错误停车，单纯max会放大错误启动；本轮最大曲率硬限速也未改善成功数。',
      '- **修自由生成一致性。** 速度loss作用于真实生成和学生闭环状态，处理solver内部速度状态与实际执行速度的一致性；GT只用于监督。谱CR当前主要约束路径Jacobian，不能由它推定速度稳定或全状态严格收缩。',
      '- **复测要求同时覆盖收益与退化。** 修复版应保留8条救回路线，并救回4条普通退化路线，再在独立路线和不同seed上配对复测。达到这些条件后再做220条；目前不扩大失败曲率规则或同一小头训练。', '',
      '## 完整性与产物','',
      '源checkpoint及两份隔离模型的SHA256均为20a5bbb009b1e8979b324d9dd5c80d164f9340e9c8531fda7658e24286c1a29d。32条策略在128生产forward帧、曲率策略在4生产forward帧中验证同输入路径逐位一致；5项速度接口/曲率测试与5项原PDM接口测试通过，编译与shell检查通过。', '',
      '- ../fm_tfpp_speed_closed_loop_20261007/report.md、pilot_summary.json、paired_route_scores.png：32条完整结果。',
      '- ../fm_tfpp_speed_closed_loop_20261007/incident_audit.json、terrain_collision_evidence.json：违规前逐帧证据。',
      '- ../fm_tfpp_curvature_closed_loop_20261008/report.md、summary.json、terrain23901_evidence.json、terrain24041_evidence.json：5条限速验证。',
      '- ../fm_tfpp_curvature_closed_loop_20261008/terrain_23901_replays.png、terrain_24041_replays.png：实际轨迹与FM预测路径。',
      '- ../../local_evaluation/bench2drive32_tie_spectral_epoch09_tfpp_speed_20261007/ 和 ../../local_evaluation/bench2drive5_tie_spectral_epoch09_tfpp_curvature_20261008/：完整评分、watchdog状态与原始diagnostics。', '',
      '复现分析：python scripts/pdm_lite/analyze_tfpp_speed_pilot.py --compact；python scripts/pdm_lite/analyze_curvature_pilot.py；python scripts/pdm_lite/report_automated_fm_diagnosis.py。']
    (dst/'report.md').write_text('\n'.join(lines)+'\n');(dst/'summary.json').write_text(json.dumps({'pilot32':{k:v for k,v in a.items() if k!='pairs'},'pilot5':{k:v for k,v in b.items() if k!='pairs'},'decision':'keep default planner; do not expand this failed curvature rule; prioritize path/interaction student-state repair'},ensure_ascii=False,indent=2)+'\n')
    print(dst/'report.md')


if __name__=='__main__':main()
