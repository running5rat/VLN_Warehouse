"""Native PDF panels: recorded warehouse results and explicitly simulated FMAE."""
import math
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image


def draw_demo(d, root, width):
    c=d.canv
    assets=root/'docs/design/figures/warehouse-assets'
    frame=root/'demo/semantic_mapping/warehouse-navigation/frames/00000197'

    def photo(path,x,y,w,h,crop=None):
        with Image.open(path) as im:
            if crop: im=im.crop(crop)
            c.drawImage(ImageReader(im),x,y,width=w,height=h,preserveAspectRatio=True,anchor='c',mask='auto')

    def card(title,path,x,y,w,h,crop=None):
        d.panel(x,y,w,h)
        d.label(title,x+8,y+h-8,w-16,'diagram_title',TA_CENTER)
        photo(path,x+5,y+5,w-10,h-32,crop)

    if d.kind=='annotation':
        import json, html
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import ParagraphStyle
        evidence=json.loads((root/'docs/design/r2r-annotation-example.json').read_text())
        left=(width-12)/2; right=left+12
        d.label('教师标注 · Qwen Plus · 一次请求',0,537,width,'diagram_title',TA_CENTER)
        def text_panel(title,body,x,y,w,h,mono=False):
            d.panel(x,y,w,h)
            d.label(title,x+10,y+h-9,w-20,'diagram_title',TA_LEFT)
            style=ParagraphStyle('annotation_body',fontName='Mono' if mono else 'Body',
                                 fontSize=7.3,leading=10,textColor=colors.HexColor('#25364A'))
            text=Paragraph(html.escape(body).replace('\n','<br/>'),style)
            _,actual=text.wrap(w-20,h-40)
            if actual>h-40:
                raise ValueError(f'{title} text needs {actual}pt, available {h-40}pt')
            text.drawOn(c,x+10,y+h-35-actual)
        text_panel('输入 · SYSTEM',evidence['instructions'],0,36,left,474)
        text_panel('输入 · USER',evidence['user_text'],right,364,left,146)
        d.panel(right,196,left,156)
        d.label('输入 · 图像序列',right+10,343,left-20,'diagram_title',TA_LEFT)
        labels=['RGB t−3 s','RGB t−2 s','RGB t−1 s','RGB t','Semantic map','Detections']
        cw=(left-24)/3
        for i,asset in enumerate(evidence['figure_images']):
            x=right+6+(i%3)*(cw+6);y=275-(i//3)*62
            photo(root/asset['path'],x,y,cw,46)
            d.label(labels[i],x,y-2,cw,'diagram',TA_CENTER)
        result='{\n'+',\n'.join('  '+json.dumps(k)+': '+json.dumps(v,ensure_ascii=False) for k,v in evidence['output'].items())+'\n}'
        text_panel('输出 · JSON',result,right,36,left,148,mono=True)
        d.label('Landmark / Local / Route → 审核标签；Waypoint / Reference GT 来自示范轨迹',0,23,width,'diagram',TA_CENTER)
        return

    if d.kind=='mapping':
        w=(width-18)/2
        card('RGB 候选帧示例',frame/'rgb.png',0,342,w,178)
        card('实例 mask · sign / storage rack',frame/'detections.png',w+18,342,w,178)
        d.label('Landmark 选择 → 匹配 mask → 投影、关联与文字绑定',0,332,width,'diagram_title',TA_CENTER)
        card('语义建图结果 · original-20261013',root/'demo/semantic_mapping/original-20261013/semantic_bev.png',0,0,width,306)
        return

    if d.kind=='gsplat':
        w=(width-18)/2
        pair=assets/'gsplat-heldout.png'
        with Image.open(pair) as im: iw,ih=im.size
        card('留出视角 · 采集 RGB',pair,0,200,w,180,(0,0,iw//2,ih))
        card('同位姿 · 3DGS 渲染',pair,w+18,200,w,180,(iw//2,0,iw,ih))
        card('新视角 A',assets/'gsplat-novel-left.png',0,0,w,180)
        card('新视角 B',assets/'gsplat-novel-right.png',w+18,0,w,180)
        d.label('demo 场景 · 70 个训练视角 / 11 个留出视角',0,195,width,'diagram',TA_CENTER)
        return

    if d.kind=='fmae':
        left=170; gx=205; gw=width-gx
        d.label('多方向当前观测',0,308,left,'diagram_title',TA_CENTER)
        photo(frame/'rgb.png',0,162,left,128)
        d.label('Front',0,160,left,'diagram',TA_CENTER)
        photo(assets/'left.png',0,81,81,61)
        photo(assets/'right.png',89,81,81,61)
        d.label('Left',0,79,81,'diagram',TA_CENTER)
        d.label('Right',89,79,81,'diagram',TA_CENTER)
        d.label('＋参考线、语义条件、机器人状态',0,58,left,'diagram',TA_CENTER)
        d.arrow(175,177,201,177)
        d.label('模拟候选 · 3 s / 30 点',gx,308,gw,'diagram_title',TA_CENTER)
        cw=(gw-12)/3; ch=62
        mode_colors=['#7356A6','#258FA4','#D58828']
        for j,title in enumerate(('LEFT','CENTER','RIGHT')):
            d.label(title,gx+j*(cw+6),288,cw,'diagram',TA_CENTER)
        for i,(name,velocity) in enumerate((('STOP-YIELD',None),('FOLLOW',.18),('PROCEED',.30))):
            y=211-i*73
            for j,side in enumerate((1,0,-1)):
                x=gx+j*(cw+6)
                d.panel(x,y,cw,ch,colors.HexColor('#F4F7FB'))
                chosen=(i==1 and j==1)
                if chosen:
                    c.setStrokeColor(colors.HexColor('#E19A28'));c.setLineWidth(1.8)
                    c.roundRect(x,y,cw,ch,7,stroke=1,fill=0)
                ox=x+cw/2;oy=y+7
                # All cells share the same body-frame scale; dashed line is nominal reference.
                c.setDash(2,2);c.setLineWidth(.7);c.setStrokeColor(colors.HexColor('#ABB7C5'))
                c.line(ox,oy,ox,y+53);c.setDash()
                pts=[(ox,oy)]
                for k in range(1,31):
                    t=k*.1
                    forward=.06*(1-math.exp(-4*t)) if velocity is None else velocity*t
                    lateral=side*.22*(forward/.9)**2
                    pts.append((ox-lateral*100,oy+forward*50))
                c.setStrokeColor(colors.HexColor(mode_colors[j]));c.setLineWidth(1.8)
                for a,b in zip(pts,pts[1:]):c.line(*a,*b)
                c.setFillColor(colors.HexColor(mode_colors[j]));c.circle(*pts[-1],2,fill=1,stroke=0)
                c.setFillColor(colors.HexColor('#21394D'));c.circle(ox,oy,1.7,fill=1,stroke=0)
                d.label(name,x,y-2,cw,'diagram',TA_CENTER)
        d.label('上：前进 x　左：车体 y　虚线：名义参考线',gx,49,gw,'diagram',TA_CENTER)
        d.panel(0,0,width,30,colors.HexColor('#FFF3DF'))
        d.label('九模态 → Scorer → 所选 traj（CENTER / FOLLOW）→ Controller',8,22,width-16,'diagram_title',TA_CENTER)
