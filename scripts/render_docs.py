"""Render the documentation's SVG diagrams with Matplotlib (report tooling only)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parents[1]/'skills/vvip/references/assets'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path','mathtext.fontset':'stix'})
INK='#171b24';MUTED='#566173';GOLD='#a06d13';BORDER='#e1d8c4';PAPER='#fffdf8'
def canvas(height,title,subtitle):
    fig,ax=plt.subplots(figsize=(12,height));fig.subplots_adjust(0,0,1,1)
    fig.patch.set_facecolor(PAPER);ax.set(xlim=(0,1200),ylim=(0,height*100));ax.axis('off')
    ax.text(36,height*100-35,title,fontsize=17,weight='bold',color=INK,va='center')
    ax.text(36,height*100-62,subtitle,fontsize=10,color=MUTED,va='center')
    return fig,ax

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(name+'.svg');fig.savefig(path,facecolor=PAPER,metadata={'Date':None})
    path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
    preview=Path(__file__).resolve().parents[1]/'artifacts'/('preview-'+name+'.png')
    preview.parent.mkdir(exist_ok=True);fig.savefig(preview,dpi=120,facecolor=PAPER);plt.close(fig)

def box(ax,x,y,w,h,title,detail,accent=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',linewidth=1.2,
                              edgecolor=GOLD if accent else BORDER,facecolor='#fff3d6' if accent else 'white'))
    ax.text(x+w/2,y+h*.67,title,ha='center',va='center',fontsize=11,weight='bold',color=INK)
    ax.text(x+w/2,y+h*.30,detail,ha='center',va='center',fontsize=9,color=MUTED,linespacing=1.4)
def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=GOLD,connectionstyle='arc3,rad=0'))

fig,ax=canvas(3.5,'Priority at the iteration boundary','The engine owns request state. VVIP selects one eligible lower-priority victim.')
box(ax,35,135,190,84,'VIP arrives','Trusted priority input')
box(ax,275,135,220,84,'Safe boundary','Pressure + policy guards',True)
box(ax,550,200,245,75,'Recompute','Free state; retain history',True)
box(ax,550,80,245,75,'Abort','Free state; end request')
box(ax,855,135,305,84,'Native vLLM scheduling','Serve VIP; resume or emit terminal')
for a,b in [((225,177),(275,177)),((495,177),(550,235)),((495,177),(550,115)),((795,235),(855,177)),((795,115),(855,177))]:arrow(ax,a,b)
ax.text(36,29,'No eligible victim? Continue native scheduling. Recompute preserves the original stream; abort is permanent.',fontsize=9,color=MUTED)
save(fig,'preemption-flow')

fig,ax=canvas(3.8,'State reconstruction and replay accounting','Semantic reconstruction is distinct from bitwise GPU reproducibility.')
ax.text(40,268,'01  REBUILD STATE',fontsize=10,color=GOLD,weight='bold')
ax.text(40,211,r'$s_0=0,\qquad s_t=F(s_{t-1},x_t)$',fontsize=26,color=INK)
ax.text(650,220,'Replay the retained token history from position zero.\nGDN + attention; prefix state caching disabled.',fontsize=11,color=MUTED,linespacing=1.8)
ax.plot([36,1164],[164,164],color=BORDER,lw=1)
ax.text(40,134,'02  COUNT ONLY REPLAYED POSITIONS',fontsize=10,color=GOLD,weight='bold')
ax.text(40,77,r'$r_k=\max\!\left(0,\,\min(e_k,w)-\max(0,e_k-n_k)\right)$',fontsize=23,color=INK)
ax.text(40,28,'w: preemption watermark     e: computed end position     n: scheduled positions     r: overlap with retained history',fontsize=10,color=MUTED)
save(fig,'state-replay')

fig,ax=canvas(5.15,'Metrics that account for both service tiers','Measure the benefit to VIPs together with ordinary-request completion and useful output.')
items=[('LATENCY',r'$\mathrm{TTFT}_i=t_i^{\mathrm{first}}-t_i^{\mathrm{send}}$',
        'First-token latency from actual client dispatch.'),
       ('VIP SERVICE TARGET',r'$A_{\tau}=\frac{\sum_{i\in V}\mathbf{1}[\,i\in C\;\wedge\;t_i^{\mathrm{first}}-t_i^{\mathrm{arrival}}\leq\tau\,]}{|V|}$',
        'All submitted VIPs are the denominator; successful completion is required.'),
       ('USEFUL THROUGHPUT',r'$Q_{\mathrm{complete}}=\frac{\sum_{i\in C}N_i}{T},\qquad R_O=\frac{|C\cap O|}{|O|}$',
        'Partial aborted output is excluded from completed-request throughput.')]
for k,(title,formula,note) in enumerate(items):
    y=414-k*140
    ax.text(38,y,title,fontsize=10,color=GOLD,weight='bold')
    ax.text(38,y-58,formula,fontsize=22,color=INK)
    ax.text(38,y-91,note,fontsize=10,color=MUTED)
    if k<2:ax.plot([36,1164],[y-110,y-110],color=BORDER,lw=1)
ax.text(660,41,'V: VIPs   O: ordinary   C: completed\nN: output tokens   T: measured drain time',fontsize=9,color=MUTED,linespacing=1.6)
save(fig,'metrics')
