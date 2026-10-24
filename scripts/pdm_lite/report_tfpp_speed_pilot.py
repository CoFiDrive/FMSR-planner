"""Render reproducible development-pilot findings from completed route evidence."""
import json
from pathlib import Path
import statistics
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fmsr-pilot-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_tfpp_speed_pilot import ANALYSIS


def main():
    d=json.loads((ANALYSIS/'pilot_summary.json').read_text())
    selection=json.loads((ANALYSIS/'pilot32_selection.json').read_text())
    hard={r['route_id'] for r in selection['routes'][:selection['hard_routes']]}
    groups={}
    for name,pairs in [('全部',d['pairs']),('原失败难路线',[p for p in d['pairs'] if p['route_id'] in hard]),('原成功普通路线',[p for p in d['pairs'] if p['route_id'] not in hard])]:
        if not pairs:continue
        groups[name]={'n':len(pairs),'baseline_DS':statistics.mean(p['baseline_DS'] for p in pairs),'hybrid_DS':statistics.mean(p['hybrid_DS'] for p in pairs),'baseline_success':sum(p['baseline_success'] for p in pairs),'hybrid_success':sum(p['hybrid_success'] for p in pairs)}
    (ANALYSIS/'subgroup_metrics.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2)+'\n')
    lines=['# FM 路径 + 冻结 TF++ 原速度：闭环开发测试','',f"进度：{d['progress'][0]}/{d['progress'][1]}；完整结束：{d['complete']}。",'',
      '零训练、原 epoch09 全部权重及 FM solver 保持；只在评估配置中选择原 TF++ 速度分类分支，停车概率 >0.9 输出0，否则取类别中心概率期望。默认配置仍使用原 planner 速度。', '',
      '| 路线组 | 数量 | 原 FM DS | 新组合 DS | 原成功数 | 新成功数 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,g in groups.items():lines.append(f"| {name} | {g['n']} | {g['baseline_DS']:.4f} | {g['hybrid_DS']:.4f} | {g['baseline_success']} | {g['hybrid_success']} |")
    lines += ['',f"已完成同路线：原 SR {d['baseline_same_routes']['SR_percent']:.4f}%，新 SR {d['hybrid']['SR_percent']:.4f}%；原 RC {d['baseline_same_routes']['RC']:.4f}%，新 RC {d['hybrid']['RC']:.4f}%。救回 {d['recovered_successes']} 条，损失 {d['lost_successes']} 条。",'',
      '成功口径：Completed/Perfect 且除 min_speed 外无违规。TickRuntime 保留评分；它在本评估中代表超过4000个仿真tick，不能当作基础设施错误删除。', '',
      '## 逐路线变化', '', '| 路线 | 场景 | 原 DS | 新 DS | 新状态 |','| --- | --- | ---: | ---: | --- |']
    for p in d['pairs']:lines.append(f"| {p['route_id']} | {p['scenario']} | {p['baseline_DS']:.4f} | {p['hybrid_DS']:.4f} | {p['hybrid_status']} |")
    lines += ['', '## 已定位的问题及证据', '',
      '1. 原 FM 速度生成存在训练桥状态到自由生成的差距：相同128开发专家帧实际生成速度 MAE 0.47217 m/s，冻结 TF++ 原解码为0.15573 m/s；接近GT终点的桥状态误差很小，不能替代完整生成评价。oracle模式仍约0.4512 m/s，因此并非只修模式分类即可解决。',
      '2. 速度与路径/控制的配合是额外问题。同一输入下切换速度来源，128帧生产forward的FM路径逐位一致；车辆执行不同速度后，访问状态及后续路径会改变，离线速度指标并不保证闭环安全。',
      '3. AccidentTwoWays / 1852：原100分，新60分。碰撞前16帧 TF++≥2 m/s 而同状态FM<0.5 m/s，随后目标约13 m/s，转向仍在调整。碰撞后两分支都停车，并出现安全框制动。早启动/加速与路径避障相互影响是需要验证的假设，历史单次对照不能证明唯一因果。见 accident1852_evidence.json 和 accident1852_speed_collision.png。',
      '4. EnterActorFlow / 2201：原100分，新60分。启动阶段存在 TF++1.76 vs FM0.055 m/s，随后实际车速较快上升；碰撞瞬间两者约9.9 m/s、已经接近。不能把碰撞时一致的速度输出当作路径安全保证，也不能仅凭本轮归咎速度头。',
      '5. 碰撞后停车、stop-sign和creep safety-box制动需要单列。diagnostic中的低速度持续时间是观测证据，不等于每一段都是错误停车；关闭安全覆盖不属于有依据的改进。', '',
      '6. Accident / 2534：碰撞前约1秒两分支均要求停车，实际车速14.7 m/s并开始转向，碰撞帧实际6.8 m/s、两分支目标同时约20 m/s。几何回放显示邻车与ego后侧接触，说明还要核查减速换道净空和预测连续性；碰撞帧的跳变不能反推为碰撞发生前的原因。见collision_2534_geometry.png。',
      '7. 原checkpoint中min_speed_infractions=179.56不是179次错误停车。当前Bench2Drive按路线检查点无条件记录周围车速比事件，再按行驶公里数汇总次数，且评分类型为unused；大于100%的记录同样存在。因此不能从这个字段的名称/数量直接认定速度过慢。依据：3rd_party/Bench2Drive/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py的MinimumSpeedRouteTest，以及leaderboard/utils/statistics_manager.py的PENALTY_PERC_DICT和compute_global_statistics。', '',
      '8. 32条全部完成后的剩余堵塞为23901 InterurbanActorFlow和24041 HighwayExit；都先发生static.terrain碰撞及越线，最终停在约(-2581,5910)同一地理区域。23901存在约60秒“目标≥2 m/s、实际|速度|<0.5 m/s”连续区间，两分支都有行驶指令；不是单纯错误停车。最终安全覆盖是在车辆已经碰撞堵塞后生效，不能因此认定关闭安全覆盖能解决。速度日志是有符号纵向速度，stationary审计使用绝对值，倒退不计停车。', '',
      '## 改进方向（优先顺序）', '',
      '- 路径感知速度融合：使用TF++专用速度query/logits，同时加入FM路径曲率、避障横向变化及碰撞风险。对两分支严重分歧的启动场景核查通行条件，避免无条件覆盖。单纯取min会保留原错误停车，单纯取max会放大错误启动。',
      '- 约束启动和正加速度，保留紧急减速能力；用起步碰撞路线验证加速约束与路径转向配合，并同时确认BlockedIntersection等路线不重新超时。当前只是待测试方向，未将阈值设为默认。',
      '- 自由生成一致性训练：速度监督作用到完整rollout的真实状态，而非只对GT插值桥；若使用TF++作为速度老师，也要处理联合路径生成状态中的速度与执行速度一致性。新增学生闭环状态并取得专家/安全标签，避免仅在899专家帧上反复调参。',
      '- 新小头优先接专用速度query/logits及实际FM路径。此前路线query均值输入的零初始化头500步只改善0.58%，没有理由直接扩大同一种训练。',
      '- 在独立路线划分和不同seed上做配对复测，再决定220条评估。谱CR只约束路径Jacobian，不能据此声称速度稳定或全状态收缩；速度维约束需独立验证。', '',
      '## 验证边界', '',
      '固定32条包括原FM失败、clean-prior成功的12条难路线及双方成功的20条普通路线；clean-prior在本组全部100分属于选择条件。基线来自已有220条中的同路线历史记录，当前新策略单次运行；这是定向开发测试，不是无偏benchmark或统计显著性结论。开发路线也曾用于FM预训练。', '',
      '源checkpoint和隔离副本SHA256均应为20a5bbb009b1e8979b324d9dd5c80d164f9340e9c8531fda7658e24286c1a29d。生产128帧forward、3项速度接口测试、5项原接口测试已通过。', '',
      '复现分析：`python scripts/pdm_lite/analyze_tfpp_speed_pilot.py --compact`；生成本报告：`python scripts/pdm_lite/report_tfpp_speed_pilot.py`。机器可读逐帧证据见pilot_summary.json；原始诊断在评估输出diagnostics目录。']
    (ANALYSIS/'report.md').write_text('\n'.join(lines)+'\n')
    ps=d['pairs'];fig,ax=plt.subplots(figsize=(11,5));xs=range(len(ps));ax.bar(xs,[p['hybrid_DS']-p['baseline_DS'] for p in ps],color=['#b84c4c' if p['hybrid_DS']<p['baseline_DS'] else '#258c75' for p in ps]);ax.axhline(0,color='black',lw=.7);ax.set_xticks(list(xs),[p['route_id'].split('_')[1] for p in ps],rotation=90);ax.set_ylabel('Hybrid DS - original FM DS');ax.set_xlabel('Route ID (fixed development pilot)');ax.set_title(f"Completed {d['progress'][0]}/32; recovered {d['recovered_successes']}, lost {d['lost_successes']}");fig.tight_layout();fig.savefig(ANALYSIS/'paired_route_scores.png',dpi=160);plt.close(fig)
    print(json.dumps({'complete':d['complete'],'groups':groups},ensure_ascii=False))


if __name__=='__main__':main()
