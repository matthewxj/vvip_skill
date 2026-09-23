"""Standalone figure from the matched summary; no inference dependency."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
data=json.loads(a.summary.read_text())['scenarios']
scenarios=['spare','saturated','burst','long-prefill']
variants=['native','native-async','recompute','abort']
labels=['Native sync','Native async','Recompute','Abort']
colors=['#475569','#94a3b8','#0369a1','#b45309']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(12,8.3))
fig.subplots_adjust(top=.79,bottom=.11,hspace=.38,wspace=.22)
metrics=[('VIP TTFT p50 (seconds; log scale)',lambda s:s['vip']['ttft_s']['p50'],True),
 ('VIP first-token SLO within 1 second (%)',lambda s:100*s['vip']['slo_attainment'],False),
 ('Completed-request output throughput (tokens/s)',lambda s:s['all']['completed_request_tokens_per_s'],False),
 ('Ordinary requests completed (%)',lambda s:100*s['ordinary']['completion_rate'],False)]
x=np.arange(4);width=.19
for ax,(title,getter,log) in zip(axes.flat,metrics):
 for k,(variant,label,color) in enumerate(zip(variants,labels,colors)):
  values=[getter(data[scenario]['variants'][variant]) for scenario in scenarios]
  ax.bar(x+(k-1.5)*width,values,width,color=color,label=label,zorder=3)
 ax.set_title(title,loc='left',fontsize=12,pad=14,fontweight='bold')
 ax.set_xticks(x,['Spare','Saturated','Burst','Long prefill'])
 ax.grid(axis='y',color='#e2e8f0',zorder=0)
 if log:ax.set_yscale('log');ax.set_ylim(bottom=.03)
 elif '%' in title:ax.set_ylim(0,108)
 else:ax.set_ylim(bottom=0)
handles,_=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.895),ncol=4,frameon=False)
fig.suptitle('VVIP: latency benefits and ordinary-request costs\nQwen3.8-27B-NVFP4 · vLLM 0.30.0 · eager · 16 single-GPU replicas',fontsize=15,fontweight='bold',y=.98)
fig.text(.5,.025,'16 paired trials per variant/scenario across 4 rotations. Finite synthetic bursts; partial aborted output excluded from completed throughput.',fontsize=8.5,ha='center')
a.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.out,dpi=170,facecolor='white')
fig.savefig(a.out.with_suffix('.png'),dpi=150,facecolor='white')
