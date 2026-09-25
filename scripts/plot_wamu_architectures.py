#!/usr/bin/env python3
"""Draw conceptual architecture and cross-modal attention-mask diagrams."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path('artifacts/architecture'); OUT.mkdir(parents=True,exist_ok=True)
BLUE='#4c78a8'; ORANGE='#f28e2b'; GREEN='#59a14f'; PURPLE='#7b6fd0'; RED='#e15759'; GREY='#e8edf3'
def box(ax,x,y,w,h,text,color,fs=10):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.03',fc=color,ec='#334e68',lw=1.2))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,wrap=True)
def arrow(ax,x1,y1,x2,y2,text='',color='#334e68',style='-|>'):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=13,lw=1.5,color=color))
 if text: ax.text((x1+x2)/2,(y1+y2)/2+.04,text,ha='center',va='center',fontsize=8,color=color)
def architecture(ax):
 ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title('OpenWAM dual-system architecture overview',weight='bold',fontsize=15)
 box(ax,.04,.72,.18,.12,'Video tokens\nV',BLUE); box(ax,.04,.16,.18,.12,'Proprio/action\ntokens A',ORANGE)
 box(ax,.34,.69,.25,.18,'Video DiT\n30 transformer blocks',BLUE); box(ax,.34,.13,.25,.18,'Action DiT\nAction transformer',ORANGE)
 box(ax,.70,.69,.22,.18,'Video hidden\nrepresentation',GREEN); box(ax,.70,.13,.22,.18,'Action prediction\nhead',ORANGE)
 arrow(ax,.22,.78,.34,.78,'video stream'); arrow(ax,.22,.22,.34,.22,'action stream'); arrow(ax,.59,.78,.70,.78)
 arrow(ax,.59,.22,.70,.22)
 box(ax,.38,.43,.17,.10,'Bridge\nfeatures',PURPLE,9); arrow(ax,.50,.69,.50,.53,'cross-modal bridge',PURPLE); arrow(ax,.55,.48,.59,.28,'action reads video',PURPLE)
 ax.text(.05,.04,'Cross-attention variants: separate Video DiT + Action DiT; detach changes gradient flow through the bridge.',fontsize=9)
 ax.text(.62,.93,'Self-attention variants replace the bridge with one mixed Video+Action block.',fontsize=9,ha='center')
def masks(ax):
 ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title('Cross-modal attention visibility',weight='bold',fontsize=15)
 modes=[('action_sees_video','Video cannot see Action\nAction sees all Video',[(1,0),(0,1)]),('isolated','Video cannot see Action\nAction sees first frame only',[(1,0),(0,.35)]),('video_sees_action','Video sees Action\n(except first frame)\nAction sees first frame only',[(.7,1),(0,.35)])]
 for i,(name,desc,vals) in enumerate(modes):
  x=.04+i*.32; ax.text(x+.13,.89,name,ha='center',weight='bold',fontsize=10); ax.text(x+.13,.80,desc,ha='center',fontsize=8)
  y=.42; s=.22; ax.add_patch(Rectangle((x,y),s,s,fc='white',ec='#334e68')); ax.plot([x+s/2,x+s/2],[y,y+s],color='#334e68'); ax.plot([x,x+s],[y+s/2,y+s/2],color='#334e68')
  # top-left V->V and bottom-right A->A always visible
  ax.add_patch(Rectangle((x,y+s/2),s/2,s/2,fc=GREY,ec='none')); ax.add_patch(Rectangle((x+s/2,y),s/2,s/2,fc=GREY,ec='none'))
  if vals[0][1]: ax.add_patch(Rectangle((x+s/2,y+s/2),s/2,s/2,fc=GREEN,ec='none',alpha=vals[0][1]))
  if vals[1][0]: ax.add_patch(Rectangle((x,y),s/2,s/2,fc=ORANGE,ec='none',alpha=vals[1][0]))
  ax.text(x+s/4,y+s+.02,'Video query',ha='center',fontsize=7); ax.text(x+3*s/4,y-.045,'Action query',ha='center',fontsize=7); ax.text(x-.025,y+3*s/4,'Video key',rotation=90,ha='center',fontsize=7); ax.text(x+s+.025,y+s/4,'Action key',rotation=90,ha='center',fontsize=7)
 ax.text(.5,.10,'Green = Video→Action visibility; orange = Action→Video visibility; grey = within-stream visibility.',ha='center',fontsize=9)
def main():
 fig1=plt.figure(figsize=(12,7)); architecture(fig1.add_subplot(111)); fig1.savefig(OUT/'openwam_architecture_overview.png',dpi=300,bbox_inches='tight');
 fig2=plt.figure(figsize=(12,7)); masks(fig2.add_subplot(111)); fig2.savefig(OUT/'openwam_attention_masks.png',dpi=300,bbox_inches='tight');
 with PdfPages(OUT/'openwam_architecture.pdf') as pdf: pdf.savefig(fig1,bbox_inches='tight'); pdf.savefig(fig2,bbox_inches='tight')
 plt.close('all')
if __name__=='__main__': main()
