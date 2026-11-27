#!/usr/bin/env python3
"""Build the v0.7 design PDF from Markdown; model/Isaac dependencies untouched."""
from __future__ import annotations
import argparse
import hashlib
import html
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Flowable, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
NAVY = colors.HexColor('#142F50')
BLUE = colors.HexColor('#176FA6')
INK = colors.HexColor('#25364A')
MUTED = colors.HexColor('#52677E')
PALE = colors.HexColor('#EFF5FA')
LINE = colors.HexColor('#D7E2ED')
WIDTH, HEIGHT = A4
MARGIN = 46
BODY_W = WIDTH - 2*MARGIN
CJK_RE = re.compile(r'([\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+)')
STYLES = {}
FIGURE_WAYPOINTS = ((480,650),(480,490))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rich(text):
    """Explicit Chinese font fallback; Droid fallback has no Latin glyphs."""
    return ''.join('<font name="CJK">'+html.escape(p)+'</font>' if CJK_RE.fullmatch(p)
                   else html.escape(p) for p in CJK_RE.split(text or '') if p)


def inline(node):
    out = rich(node.text)
    for ch in node:
        body = inline(ch)
        if ch.tag in ('strong', 'b'): body = '<b>'+body+'</b>'
        elif ch.tag in ('em', 'i'): body = '<i>'+body+'</i>'
        elif ch.tag == 'code': body = '<font name="Mono" color="#234F73">'+body+'</font>'
        elif ch.tag == 'a':
            target = ch.get('href','')
            if target.startswith(('https://','http://')):
                body = '<link href="'+html.escape(target,quote=True)+'" color="#176FA6">'+body+'</link>'
        elif ch.tag == 'br': body = '<br/>'
        out += body + rich(ch.tail)
    return out


def para(text, kind='body'):
    return Paragraph(rich(text), STYLES[kind])


class Diagram(Flowable):
    """Document-native vectors with explicitly sourced demonstration photographs."""
    def __init__(self, kind):
        Flowable.__init__(self)
        self.kind=kind
        self.width=BODY_W
        self.height={'overview':BODY_W * 447 / 504,'anchors':268,'training':372,'deployment':222,
                     'mapping':520,'fmae':310,'gsplat':380,'annotation':540,
                     'demonstration':222,'memorytask':330,'staticoracle':300,'navmesh':260,'navmeshquery':260}[kind]

    def label(self,text,x,y,w,kind='diagram',align=TA_LEFT):
        style=ParagraphStyle('figure_label',parent=STYLES[kind],alignment=align)
        t=Paragraph(rich(text).replace('\n','<br/>'),style);_,h=t.wrap(w,700);t.drawOn(self.canv,x,y-h)
        return h

    def panel(self,x,y,w,h,color=PALE):
        c=self.canv;c.setFillColor(color);c.setStrokeColor(LINE)
        c.roundRect(x,y,w,h,7,fill=1,stroke=1)

    def box(self,x,y,w,h,title,desc):
        self.panel(x,y,w,h)
        self.label(title,x+8,y+h-8,w-16,'diagram_title',TA_CENTER)
        self.label(desc,x+8,y+22,w-16,'diagram',TA_CENTER)

    def arrow(self,x1,y1,x2,y2,dashed=False):
        import math
        c=self.canv;c.setStrokeColor(BLUE);c.setFillColor(BLUE);c.setLineWidth(1.15)
        c.setDash(3,2) if dashed else c.setDash()
        c.line(x1,y1,x2,y2);c.setDash()
        a=math.atan2(y2-y1,x2-x1);p=c.beginPath();p.moveTo(x2,y2)
        p.lineTo(x2-6*math.cos(a-.45),y2-6*math.sin(a-.45))
        p.lineTo(x2-6*math.cos(a+.45),y2-6*math.sin(a+.45));p.close()
        c.drawPath(p,fill=1,stroke=0)

    def photo(self,name,x,y,w,h,crop=None):
        from PIL import Image
        from reportlab.lib.utils import ImageReader
        im=Image.open(ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000197'/name)
        if crop:im=im.crop(crop)
        self.canv.drawImage(ImageReader(im),x,y,width=w,height=h,preserveAspectRatio=True,anchor='c',mask='auto')

    def draw(self):
        c=self.canv
        if self.kind in ('demonstration','memorytask','staticoracle','navmesh','navmeshquery'):
            from dataengine_figures import draw_dataengine
            draw_dataengine(self, ROOT, BODY_W)
            return
        if self.kind in ('mapping','fmae','gsplat','annotation'):
            from demo_figures import draw_demo
            draw_demo(self, ROOT, BODY_W)
            return
        if self.kind=='overview':
            # Embed the actual v0.6 vector artwork, with only the requested additions.
            from pdfrw import PdfReader
            from pdfrw.buildxobj import pagexobj
            from pdfrw.toreportlab import makerl
            asset=ROOT/'docs/design/figures/v06-enriched-overview.pdf'
            xobj=pagexobj(PdfReader(str(asset)).pages[0])
            c.saveState();c.scale(BODY_W/504,self.height/447)
            c.doForm(makerl(c,xobj));c.restoreState()
            return
        if self.kind=='anchors':
            w=(BODY_W-28)/2;right=w+28
            self.panel(0,30,w,228);self.panel(right,30,w,228)
            self.label('RGB 像素锚点',9,250,w-18,'diagram_title')
            self.photo('rgb.png',7,48,w-14,178)
            # Explicitly illustrative pixels on visible floor, not historical five-point predictions.
            from PIL import Image
            iw,ih=Image.open(ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000197/rgb.png').size
            scale=min((w-14)/iw,178/ih);ox=7+((w-14)-iw*scale)/2;oy=48+(178-ih*scale)/2
            for name,(u,v) in zip(('P1','P2'),FIGURE_WAYPOINTS):
                x=ox+u*scale;y=oy+(ih-v)*scale
                c.setFillColor(colors.HexColor('#FFB534'));c.setStrokeColor(colors.white);c.circle(x,y,4,fill=1,stroke=1)
                self.label(name,x+7,y+5,28,'diagram_title')
            self.label('近点 → 远点；均在可见地面上',9,45,w-18)
            self.label('连续参考线的 RGB 投影',right+9,250,w-18,'diagram_title')
            self.photo('rgb.png',right+7,48,w-14,178)
            ox=right+7+((w-14)-iw*scale)/2;oy=48+(178-ih*scale)/2
            c.setStrokeColor(colors.HexColor('#00BCD4'));c.setLineWidth(2)
            points=[(ox+480*scale,oy+(ih-(710-220*i/31))*scale) for i in range(32)]
            for a,b in zip(points,points[1:]):c.line(*a,*b)
            for u,v in FIGURE_WAYPOINTS:
                x=ox+u*scale;y=oy+(ih-v)*scale
                c.setFillColor(colors.HexColor('#FFB534'));c.setStrokeColor(colors.white);c.circle(x,y,4,fill=1,stroke=1)
            self.label('连续曲线与两个锚点叠加显示',right+9,45,w-18)
            self.arrow(w+2,145,right-2,145)
            self.label('REFLINE 特征 → 连续参考线 → 相机投影',0,20,BODY_W,align=TA_CENTER)
            return
        if self.kind=='training':
            import math
            warm=colors.HexColor('#C65C2D');cool=colors.HexColor('#347DB0')
            idle=colors.HexColor('#A2ACB5')
            def status(x,y,state):
                if state=='train':
                    c.setFillColor(warm);c.setStrokeColor(warm)
                    p=c.beginPath();p.moveTo(x,y-5)
                    p.curveTo(x-8,y-3,x-5,y+3,x-2,y+5)
                    p.curveTo(x-3,y+1,x+1,y+2,x+1,y+8)
                    p.curveTo(x+9,y+1,x+7,y-5,x,y-5)
                    c.drawPath(p,fill=1,stroke=0)
                elif state=='frozen':
                    c.setStrokeColor(cool);c.setLineWidth(.8)
                    for k in range(6):
                        a=k*math.pi/3;ex=x+6*math.cos(a);ey=y+6*math.sin(a)
                        c.line(x,y,ex,ey)
                        for d in (-.6,.6):
                            c.line(ex,ey,ex-2.5*math.cos(a+d),ey-2.5*math.sin(a+d))
                else:
                    c.setStrokeColor(idle);c.setLineWidth(1);c.line(x-5,y,x+5,y)
            for x,state,title in [(12,'train','参数更新'),(130,'frozen','冻结参数'),(247,'idle','未参与')]:
                status(x,360,state);self.label(title,x+13,366,90)
            self.label('分阶段继承权重',365,366,138,align=TA_CENTER)
            columns=[(88,102),(218,112),(365,128)]
            stages=[
                ('Stage1','语义学习',278,70,('train','idle','idle'),'文本＋两个锚点 · Cross-entropy'),
                ('Stage2','参考线学习',198,70,('frozen','train','idle'),'语义＋前视几何 → 路线长度／角度纠错'),
                ('Stage3','动作与评分',118,70,('frozen','frozen','train'),'Flow Matching＋Hydra 蒸馏 · 动作侧感知与条件融合'),
                ('Stage4','仓库闭环',18,90,('train','train','train'),'联合微调 · DAgger＋RL · 历史回放'),
            ]
            for n,(stage,title,y,h,states,note) in enumerate(stages):
                c.setFillColor(colors.HexColor('#F6F8FA'));c.setStrokeColor(LINE);c.setLineWidth(.6)
                c.roundRect(0,y,BODY_W,h,7,fill=1,stroke=1)
                self.label(stage,10,y+h-9,72,'diagram_title')
                self.label(title,10,y+h-27,72)
                by=y+h-40
                for j,((x,w),state) in enumerate(zip(columns,states)):
                    fill={'train':'#FCEBDD','frozen':'#E5F0F8','idle':'#F0F2F4'}[state]
                    border={'train':warm,'frozen':cool,'idle':idle}[state]
                    c.setFillColor(colors.HexColor(fill));c.setStrokeColor(border)
                    c.setLineWidth(.8);c.setDash(2,2) if state=='idle' else c.setDash()
                    c.roundRect(x,by,w,30,5,fill=1,stroke=1);c.setDash()
                    status(x+10,by+15,state)
                    name=('VLM','Reference Decoder','FMAE + Scorer')[j]
                    self.label(name,x+22,by+21,w-27,'diagram',TA_CENTER)
                if n>=1:self.arrow(190,by+15,216,by+15)
                if n>=2:self.arrow(330,by+15,363,by+15)
                if n==0:self.label('LoRA / Full SFT',88,y+23,405)
                self.label(note,88,y+11 if n==0 else y+23,405 if n<3 else 258)
                if n==3:
                    c.setFillColor(colors.HexColor('#E6F0E2'));c.setStrokeColor(colors.HexColor('#8CAA80'))
                    c.roundRect(365,y+4,128,22,4,fill=1,stroke=1)
                    self.label('Warehouse Env',370,y+20,118,'diagram',TA_CENTER)
                    self.arrow(397,by,397,y+27);self.arrow(465,y+27,465,by)
                if n<3:self.arrow(40,y-1,40,stages[n+1][2]+stages[n+1][3]+2)
            return
        if self.kind=='deployment':
            x0=88;x1=BODY_W-10;span=x1-x0
            rows=[('VLM',173),('FMAE',110),('Control',47)]
            for title,y in rows:
                t=para(title,'diagram_title');t.wrap(75,20);t.drawOn(c,0,y+8)
                c.setStrokeColor(LINE);c.line(x0,y,x1,y)
            for a,b,label in [(0,.42,'请求 k'),(.53,.94,'请求 k+1')]:
                self.box(x0+a*span,173,(b-a)*span,40,label,'推理后发布新快照')
            for a in range(8):
                x=x0+a*span/8;c.setFillColor(colors.HexColor('#C6E1F0'));c.roundRect(x,110,span/8-5,29,3,fill=1,stroke=0)
            for a in range(26):
                x=x0+a*span/26;c.setFillColor(colors.HexColor('#176FA6'));c.rect(x,47,span/26-3,18,fill=1,stroke=0)
            for a in (.42,.94):self.arrow(x0+a*span,173,x0+a*span,139,True)
            self.arrow(x0,24,x1,24)
            t=para('时间 →  /  快控制持续；语义更新后对齐接管，不等待旧路线耗尽。','diagram');t.wrap(BODY_W,20);t.drawOn(c,0,1)
            return

class DesignDoc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
                         topMargin=53,bottomMargin=44,allowSplitting=True,
                         title='动态仓库 VLN 系统设计文档 v0.7',author='VLN Warehouse project',
                         subject='Four-part revision: pipeline, training, data engine and deployment',
                         pageCompression=1)
        f=Frame(MARGIN,44,BODY_W,HEIGHT-97,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='normal',frames=[f],onPage=self.header))

    def header(self,c,doc):
        if doc.page==1:return
        c.saveState();c.setStrokeColor(LINE);c.line(MARGIN,HEIGHT-37,WIDTH-MARGIN,HEIGHT-37)
        c.setFont('Body',8);c.setFillColor(MUTED)
        c.drawString(MARGIN,HEIGHT-27,'DYNAMIC WAREHOUSE VLN  /  v0.7')
        c.drawRightString(WIDTH-MARGIN,HEIGHT-27,'2026-11-27')
        c.line(MARGIN,34,WIDTH-MARGIN,34)
        c.setFont('Body',7.5);c.drawString(MARGIN,22,'MODEL / TRAINING / DATA / DEPLOYMENT')
        c.drawRightString(WIDTH-MARGIN,22,str(doc.page))
        c.restoreState()

    def afterFlowable(self,flow):
        if isinstance(flow,Paragraph) and hasattr(flow,'bookmark_key'):
            text=flow.getPlainText();key=flow.bookmark_key;level=flow.outline_level
            self.canv.bookmarkHorizontalAbsolute(key,self.frame._y+flow.height+flow.getSpaceAfter())
            self.canv.addOutlineEntry(text,key,level=level,closed=level>0)
            if flow.in_toc:self.notify('TOCEntry',(level,rich(text),self.page,key))


def build_fonts(cjk):
    paths={'Body':Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
           'BodyBold':Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
           'BodyItalic':Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'),
           'Mono':Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'),'CJK':cjk}
    for name,path in paths.items():pdfmetrics.registerFont(TTFont(name,str(path)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    pdfmetrics.registerFontFamily('Mono',normal='Mono',bold='Mono',italic='Mono',boldItalic='Mono')
    pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJK',italic='CJK',boldItalic='CJK')
    base=dict(fontName='Body',fontSize=9.4,leading=14.5,textColor=INK,wordWrap='CJK',spaceAfter=5,splitLongWords=True)
    STYLES['body']=ParagraphStyle('body',**base)
    STYLES['h2']=ParagraphStyle('h2',parent=STYLES['body'],fontName='BodyBold',fontSize=19,leading=27,textColor=NAVY,spaceAfter=16,keepWithNext=True)
    STYLES['h3']=ParagraphStyle('h3',parent=STYLES['body'],fontName='BodyBold',fontSize=12.5,leading=20,textColor=BLUE,spaceBefore=9,spaceAfter=6,keepWithNext=True)
    STYLES['h4']=ParagraphStyle('h4',parent=STYLES['body'],fontName='BodyBold',fontSize=10.4,leading=17,spaceBefore=9,spaceAfter=6,keepWithNext=True)
    STYLES['h5']=ParagraphStyle('h5',parent=STYLES['h4'],fontSize=9.8,leading=16)
    STYLES['cell']=ParagraphStyle('cell',parent=STYLES['body'],fontSize=8.5,leading=13.0,spaceAfter=0)
    STYLES['th']=ParagraphStyle('th',parent=STYLES['cell'],fontName='BodyBold',textColor=colors.white)
    STYLES['code']=ParagraphStyle('code',parent=STYLES['body'],fontName='Mono',fontSize=7.6,leading=11.6,spaceAfter=0)
    STYLES['diagram']=ParagraphStyle('diagram',parent=STYLES['body'],fontSize=7.4,leading=11,spaceAfter=0,alignment=TA_CENTER)
    STYLES['diagram_title']=ParagraphStyle('diagram_title',parent=STYLES['diagram'],fontName='BodyBold',fontSize=9.8,leading=14,textColor=NAVY)
    STYLES['toc']=ParagraphStyle('toc',parent=STYLES['body'],fontName='BodyBold',fontSize=11,leading=17,spaceBefore=7,spaceAfter=3)
    STYLES['toc1']=ParagraphStyle('toc1',parent=STYLES['body'],fontSize=9.2,leading=14,leftIndent=14,spaceAfter=2)
    STYLES['toc2']=ParagraphStyle('toc2',parent=STYLES['body'],fontSize=8.5,leading=13,leftIndent=28,spaceAfter=1)
    STYLES['toc3']=ParagraphStyle('toc3',parent=STYLES['toc2'],fontSize=8.1,leftIndent=42)
    return paths


def cover():
    title=ParagraphStyle('cover_title',parent=STYLES['body'],fontSize=32,leading=43,textColor=NAVY)
    sub=ParagraphStyle('cover_sub',parent=STYLES['body'],fontSize=12,leading=20,textColor=MUTED)
    flow=[Spacer(1,96),para('DYNAMIC WAREHOUSE VLN  /  v0.7','h3'),Spacer(1,14),
          Paragraph(rich('动态仓库 VLN')+'<br/>'+rich('系统原理与算法方案'),title),Spacer(1,23),
          Paragraph('Model Pipeline · Training Schedule<br/>Data Engine · Deployment Optimization',sub),
          Spacer(1,25),para('2026-11-27  |  算法原理与方案','h3'),
          para('当前开发：RTX 4090 · Isaac Sim 5.0 · 单机器人 RGB-D / 仿真 odom'),
          para('目标部署：NVIDIA Jetson AGX Orin 64GB + 差速轮式机器人'),Spacer(1,23)]
    t=Table([[para('前视VLM与地图理解，多方向AE与快速控制。保留两个RGB锚点和连续参考线；以LoRA、模型容量对照与有门槛全量SFT组织训练，以可核验轨迹和递归语义构建数据。')]],colWidths=[BODY_W])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.5,LINE),('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),('TOPPADDING',(0,0),(-1,-1),13),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    flow += [t,Spacer(1,20),para('面向仓库地标导航、探索与跟随的分层视觉语言动作架构。','body'),PageBreak()]
    return flow


def parse_story(source):
    text=source.split('<!-- cover-end -->',1)[1]
    # Bare protocol tags in prose are text, not HTML. Preserve fenced/inline code.
    prepared=[];fenced=False
    for line in text.splitlines():
        if line.startswith('```'):fenced=not fenced
        if not fenced and not line.startswith(('```','<!--')):
            chunks=re.split(r'(`[^`]+`)',line)
            line=''.join(chunk if chunk.startswith('`') else chunk.replace('<','&lt;').replace('>','&gt;') for chunk in chunks)
        prepared.append(line)
    text='\n'.join(prepared)
    text=text.replace('<!-- toc -->','<toc />')
    text=re.sub(r'<!-- diagram:(\w+) -->',r'<diagram kind="\1" />',text)
    doc=ET.fromstring('<document>'+markdown.markdown(text,extensions=['tables','fenced_code','sane_lists'])+'</document>')
    story=cover();count=0;chapter_seen=False;sub_index=0
    nodes=list(doc);skip_next=False
    for index,node in enumerate(nodes):
        if skip_next:
            skip_next=False
            continue
        if node.tag == 'p' and len(node) == 1 and node[0].tag in ('toc', 'diagram'):
            node = node[0]
        tag=node.tag
        if tag in ('h2','h3','h4','h5'):
            txt=''.join(node.itertext());is_chapter=tag=='h2' and bool(re.match(r'^\d+\.',txt))
            if is_chapter:story.append(PageBreak());chapter_seen=True
            h=Paragraph(inline(node),STYLES[tag]);h.bookmark_key=f'section-{count}';count+=1
            h.outline_level=int(tag[1])-2
            h.in_toc=bool(re.match(r'^\d+(?:\.\d+)*[. ]',txt))
            story.append(h)
        elif tag=='p':story.append(Paragraph(inline(node),STYLES['body']))
        elif tag=='toc':
            if not isinstance(story[-1],PageBreak):story.append(PageBreak())
            story.append(para('目录','h2'))
            toc=TableOfContents();toc.levelStyles=[STYLES['toc'],STYLES['toc1'],STYLES['toc2'],STYLES['toc3']];toc.dotsMinLevel=0
            story.append(toc);story.append(Spacer(1,20))
            story.append(para('目录包含全部编号标题；各条目及 PDF 书签均可跳转。正文聚焦算法分工与学习方案。'))
        elif tag=='diagram':
            figure=[Diagram(node.get('kind')),Spacer(1,6)]
            if story and hasattr(story[-1], 'bookmark_key'):
                figure.insert(0, story.pop())
            if index+1<len(nodes) and nodes[index+1].tag=='p' and ''.join(nodes[index+1].itertext()).startswith('图 '):
                figure.append(Paragraph(inline(nodes[index+1]),STYLES['body']))
                skip_next=True
            story.append(KeepTogether(figure))
        elif tag=='pre':
            raw=''.join(node.itertext()).rstrip('\n')
            lines=[rich(line).replace('  ','&#160;&#160;') or '&#160;' for line in raw.splitlines()]
            block=Paragraph('<br/>'.join(lines),STYLES['code'])
            table=Table([[block]],colWidths=[BODY_W])
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('LINEBEFORE',(0,0),(0,-1),2,BLUE),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
            story.extend([table,Spacer(1,9)])
        elif tag in ('ul','ol'):
            for i,li in enumerate(node.findall('li')):
                prefix=f'{i+1}. ' if tag=='ol' else '• '
                # Markdown list items here are flat prose, no nested blocks.
                story.append(Paragraph(rich(prefix)+inline(li),STYLES['body']))
        elif tag=='table':
            rows=[]
            for ri,row in enumerate(node.findall('.//tr')):
                rows.append([Paragraph(inline(cell),STYLES['th' if ri==0 else 'cell']) for cell in row])
            n=len(rows[0]);weights={3:[.85,1.7,1.65],4:[.75,1.2,1.3,1.1]}.get(n,[1]*n)
            widths=[BODY_W*w/sum(weights) for w in weights]
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,NAVY),('LINEBELOW',(0,1),(-1,-1),.35,LINE),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            if rows[0][0].getPlainText()=='模块':
                story.extend([KeepTogether([table]),Spacer(1,10)])
            else:
                story.extend([table,Spacer(1,10)])
        else:raise ValueError(f'Unsupported Markdown node: {tag}')
    return story


def expand_toc_links(path):
    """Make the full printed TOC row clickable, with heading-level destinations."""
    import pymupdf
    pdf=pymupdf.open(path)
    entries=[e for e in pdf.get_toc(simple=False) if re.match(r'^\d+\.', e[1])]
    dest={e[1].split()[0]:e for e in entries}
    first_body=min(e[2] for e in entries)-1
    count=0
    for pnum in range(1,first_body):
        page=pdf[pnum]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                label=''.join(span['text'] for span in line['spans']).strip()
                m=re.match(r'^(\d+(?:\.\d+)*\.?)(?=\s)',label)
                if not m or m.group(1) not in dest:continue
                e=dest[m.group(1)];r=pymupdf.Rect(line['bbox'])
                # Covers label, leaders and page number for PDF.js and desktop viewers.
                r.x0=MARGIN;r.x1=WIDTH-MARGIN;r.y0-=1;r.y1+=1
                page.insert_link({'kind':pymupdf.LINK_GOTO,'from':r,'page':e[2]-1,'to':e[3].get('to',pymupdf.Point(0,0))})
                count+=1
    if count!=len(entries):raise AssertionError(f'TOC row links {count} != numbered headings {len(entries)}')
    tmp=path.with_suffix('.linked.pdf');pdf.save(tmp,garbage=4,deflate=True);pdf.close();tmp.replace(path)
    return count


def prepare_v06_overview():
    """Copy v0.6 figure 1 vector-for-vector; augment only requested image/input sites."""
    import math
    import pymupdf
    directory=ROOT/'docs/design/figures';directory.mkdir(exist_ok=True)
    original=pymupdf.open(ROOT/'dynamic-warehouse-vln-design-v0.6.pdf')
    work=pymupdf.open();work.insert_pdf(original,from_page=3,to_page=3)
    page=work[0]
    # Cropping only hides source-page text; PDF viewers may still select it.
    # Remove text outside the figure before embedding the vector artwork.
    figure_clip=pymupdf.Rect(76,269,580,716)
    for rect in (pymupdf.Rect(0,0,page.rect.width,figure_clip.y0),
                 pymupdf.Rect(0,figure_clip.y1,page.rect.width,page.rect.height),
                 pymupdf.Rect(0,figure_clip.y0,figure_clip.x0,figure_clip.y1),
                 pymupdf.Rect(figure_clip.x1,figure_clip.y0,page.rect.width,figure_clip.y1)):
        page.add_redact_annot(rect,fill=False)
    page.apply_redactions(images=0,graphics=0)
    # Replace only the labels that need space for photographs or the multi-view input.
    for phrase in ('Source RGB','Semantic Annotation','Segmentation network',
                   'Current images / BEV / robot state','Current state / obstacles',
                   '+ Safety Shield','+ Mode Selector','Map Fusion','Depth + calibration','Vision + WS + IMU',
                   'Align + associate','Target loss or stale guidance','Historical / peer maps',
                   'Geometry Fusion','Point cloud / BEV','Semantic Projection','Landmark + Local',
                   'Generation · non-follow tasks','Updated Route','Reference features','Semantic Map'):
        for rect in page.search_for(phrase):page.add_redact_annot(rect,fill=False)
    for rect in page.search_for('Target loss'):
        page.add_redact_annot(pymupdf.Rect(280,rect.y0,455,rect.y1),fill=False)
    for rect in page.search_for('Controller'):
        if rect.y0>650:page.add_redact_annot(rect,fill=False)
    page.apply_redactions(images=0,graphics=0)
    # Remove the complete state/obstacle arrow paths, preserving the controller border.
    page.add_redact_annot(pymupdf.Rect(369,673,407,683),fill=False)
    page.apply_redactions(images=0,graphics=1)
    # VIO is shown as a module without an incoming source label or arrow.
    page.add_redact_annot(pymupdf.Rect(160,327,174,337),fill=False)
    page.apply_redactions(images=0,graphics=1)
    # Remove complete original paths including stroke extents, then redraw once.
    # The enclosing lane and VLM borders extend outside this rectangle and survive.
    page.add_redact_annot(pymupdf.Rect(398,270,520,477),fill=False)
    page.apply_redactions(images=0,graphics=1,text=1)
    # Remove annotation backdrop, old shared semantic tree and recursive-input layout.
    for rect in ((169,382,271,420),(145,390,182,412),(213,411,359,478),
                 (125,433,200,493),(70,486,238,556)):
        page.add_redact_annot(pymupdf.Rect(*rect),fill=False)
    page.apply_redactions(images=0,graphics=1,text=1)
    page.add_redact_annot(pymupdf.Rect(80,430,214,516),fill=False)
    page.apply_redactions(images=0,graphics=0)
    purple=(.47,.31,.66);teal=(0,.51,.47);blue=(.13,.43,.72);orange=(.82,.44,.13)
    assets=ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000197'
    def label(x,y,text,size=7,color=purple,bold=False):
        page.insert_text((x,y),text,fontsize=size,fontname='hebo' if bold else 'helv',color=color)
    def photograph(name,rect):
        path=Path(name) if Path(name).is_absolute() else assets/name
        pix=pymupdf.Pixmap(str(path))
        scale=min(rect.width/pix.width,rect.height/pix.height)
        w,h=pix.width*scale,pix.height*scale
        fitted=pymupdf.Rect(rect.x0+(rect.width-w)/2,rect.y0+(rect.height-h)/2,rect.x0+(rect.width+w)/2,rect.y0+(rect.height+h)/2)
        page.draw_rect(fitted+(-1,-1,1,1),color=(.70,.78,.84),fill=(1,1,1),width=.5)
        page.insert_image(fitted,filename=str(path),keep_proportion=True)
        return fitted
    def centered(rect,text,size,color):
        font=pymupdf.Font('hebo')
        x=(rect.x0+rect.x1-font.text_length(text,fontsize=size))/2
        y=(rect.y0+rect.y1)/2+(font.ascender+font.descender)*size/2
        label(x,y,text,size,color=color,bold=True)
    def connector(points,color=teal,dashed=False):
        page.draw_polyline(points,color=color,width=.9,dashes='[3 2] 0' if dashed else None)
        a,b=points[-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0]);length=4
        wings=[(b[0]-length*math.cos(angle+t),b[1]-length*math.sin(angle+t)) for t in (-.45,.45)]
        page.draw_polyline([wings[0],b,wings[1]],color=color,width=.9)
    centered(pymupdf.Rect(300,293,403,323),'SLAM',10,teal)
    centered(pymupdf.Rect(300,386,403,417),'Semantic Projection',9.2,purple)
    fusion=pymupdf.Rect(437,350,507,384)
    page.draw_rect(fusion,color=(.55,.75,.73),fill=(.94,.97,.97),width=.6,radius=.12)
    centered(fusion,'Map Fusion',9,teal)
    connector([(403,309),(419,309)])
    connector([(467.5,335),(467.5,350)])
    connector([(403,400),(420,400),(420,375),(437,375)],purple)
    connector([(551,344),(520,344),(520,365),(507,365)])
    label(522,339,'Historical maps',6.3,color=(.30,.39,.47))
    connector([(472,384),(472,475)])
    label(434,678,'Controller',9,color=teal,bold=True)
    label(432,605,'+ Hydra heads',7.5,color=(.30,.39,.47))
    label(102,378,'Source RGB',7)
    source_rect=photograph('rgb.png',pymupdf.Rect(89,382,161,423))
    label(174,378,'Semantic Annotation',7,bold=True)
    annotation_rect=photograph('detections.png',pymupdf.Rect(174,382,266,423))
    connector([(source_rect.x1,401),(annotation_rect.x0,401)],purple)
    page.draw_line((annotation_rect.x1,401.3),(267.1,401.3),color=purple,width=.9)
    # Observed detections enter VLM; generated Landmark returns separately.
    connector([(220,annotation_rect.y1),(220,431),(248,431),(248,475)],blue)
    connector([(352,475),(352,456)],purple,True)
    connector([(352,438),(352,417)],purple,True)
    generated=directory/'warehouse-assets'
    # Result example beside the outgoing Semantic Map edge, outside Map Fusion.
    photograph(ROOT/'demo/semantic_mapping/original-20261013/semantic_bev.png',pymupdf.Rect(478,405,575,468))
    photograph(generated/'depth.png',pymupdf.Rect(89,291,161,338))
    photograph(generated/'global-occupancy.png',pymupdf.Rect(419,272,516,335))
    label(410,421,'Semantic Map',7.8,color=teal)
    # Preserve the source semantic text; adapt figure anchors to the two-point design.
    output=json.loads((assets/'parsed.json').read_text())
    def callout(rect,text,size=6.2):
        page.draw_rect(rect,color=(.79,.72,.87),fill=(.98,.97,1),width=.5,radius=.08)
        remaining=page.insert_textbox(rect+(3,3,-3,-3),text,fontsize=size,fontname='helv',color=purple,lineheight=1.15)
        if remaining<0:raise ValueError('Output callout overflow: '+text)
    callout(pymupdf.Rect(307,438,466,456),'<landmark> '+json.dumps(output['landmark'])+' </landmark>')
    label(92,431,'Initial instruction',7,color=(.14,.24,.33),bold=True)
    connector([(145,432),(145,438)],color=(.14,.24,.33))
    callout(pymupdf.Rect(83,438,212,480),'<Route> '+output['route']+' </Route>',5.8)
    callout(pymupdf.Rect(83,485,212,511),'<local> '+output['local']+' </local>',6.1)
    # Both generated fields become next-request inputs, with separate input arrows.
    connector([(232,539),(232,552),(77,552),(77,458),(83,458)],purple,True)
    connector([(77,498),(83,498)],purple,True)
    connector([(212,458),(215,458),(215,489),(219,489)],purple)
    connector([(212,498),(219,510)],purple)
    label(92,532,'Images + robot state',7,color=(.30,.39,.47))
    connector([(191,529),(219,529)],blue)
    label(235,565,'Reference features',5,color=orange)
    callout(pymupdf.Rect(233,520,500,536),'<waypoint> '+ '; '.join(f'({u},{v})' for u,v in FIGURE_WAYPOINTS)+' </waypoint>',6.1)
    # A concrete warehouse route illustration beside the unchanged decoder output edge.
    # This is an explanatory overlay, never attributed to an actual decoder inference.
    label(86,621,'Reference line',6.7,color=orange,bold=True)
    rgb=pymupdf.Rect(85,626,195,682)
    photograph('rgb.png',rgb)
    iw,ih=960,720
    # Derive dimensions from the same original image; no artificial multiview photographs.
    pix=pymupdf.Pixmap(str(assets/'rgb.png'));iw,ih=pix.width,pix.height
    scale=min(rgb.width/iw,rgb.height/ih)
    ox=rgb.x0+(rgb.width-iw*scale)/2;oy=rgb.y0+(rgb.height-ih*scale)/2
    pts=[pymupdf.Point(ox+480*scale,oy+(710-220*i/31)*scale) for i in range(32)]
    page.draw_polyline(pts,color=(0,.72,.88),width=1.25)
    for u,v in FIGURE_WAYPOINTS:
        page.draw_circle((ox+u*scale,oy+v*scale),1.5,color=(1,.66,.12),fill=(1,.66,.12))
    page.draw_polyline([(209,599),(209,617),(180,617),(180,624)],color=orange,width=.6)
    page.draw_polyline([(177,621),(180,624),(183,621)],color=orange,width=.6)
    # Real Isaac camera re-renders at the same recorded body pose, +/-90 degrees.
    label(225,648,'Multi-view RGB / BEV / robot state',7.8,color=(.30,.39,.47))
    label(250,658,'Left',6,color=blue)
    label(335,658,'Right',6,color=blue)
    photograph(generated/'left.png',pymupdf.Rect(223,661,301,711))
    photograph(generated/'right.png',pymupdf.Rect(309,661,387,711))
    # Original diagram crop; only the requested controller side-input is removed.
    out=pymupdf.open();dst=out.new_page(width=504,height=447)
    dst.show_pdf_page(dst.rect,work,0,clip=figure_clip)
    dest=directory/'v06-enriched-overview.pdf';out.save(dest,garbage=4,deflate=True)
    out.close();work.close();original.close()
    return dest


def export_architecture():
    from reportlab.pdfgen.canvas import Canvas
    import pymupdf
    directory=ROOT/'docs/design/figures';directory.mkdir(exist_ok=True)
    path=directory/'model-pipeline-v07.pdf'
    d=Diagram('overview');c=Canvas(str(path),pagesize=(BODY_W+20,d.height+20))
    d.drawOn(c,10,10);c.showPage();c.save()
    pdf=pymupdf.open(path);pdf[0].get_pixmap(matrix=pymupdf.Matrix(3,3)).save(directory/'model-pipeline-v07.png');pdf.close()
    return {str(p.relative_to(ROOT)):sha(p) for p in [path,directory/'model-pipeline-v07.png']}


def export_training():
    from reportlab.pdfgen.canvas import Canvas
    import pymupdf
    directory=ROOT/'docs/design/figures'
    path=directory/'training-schedule-v07.pdf'
    diagram=Diagram('training')
    canvas=Canvas(str(path),pagesize=(BODY_W+20,diagram.height+20))
    diagram.drawOn(canvas,10,10);canvas.showPage();canvas.save()
    pdf=pymupdf.open(path)
    png=path.with_suffix('.png')
    pdf[0].get_pixmap(matrix=pymupdf.Matrix(3,3)).save(png);pdf.close()
    return {str(p.relative_to(ROOT)):sha(p) for p in (path,png)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'dynamic-warehouse-vln-design-v0.7.md')
    ap.add_argument('--output',type=Path,default=ROOT/'dynamic-warehouse-vln-design-v0.7.pdf')
    ap.add_argument('--cjk-font',type=Path)
    args=ap.parse_args()
    candidates=[args.cjk_font,Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),Path('/proc/1/root/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')]
    font=next((p for p in candidates if p and p.is_file()),None)
    if font is None:raise SystemExit('Provide --cjk-font with a Chinese TrueType font (Droid Sans Fallback).')
    fonts=build_fonts(font)
    prepare_v06_overview()
    text=args.source.read_text();story=parse_story(text)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    DesignDoc(args.output).multiBuild(story)
    toc_link_rows=expand_toc_links(args.output)
    architecture_files=export_architecture()
    training_files=export_training()
    from reportlab.pdfgen.canvas import Canvas
    import pymupdf
    demo_files={}
    for kind in ('mapping','anchors','fmae','gsplat','annotation','demonstration','memorytask','navmesh','navmeshquery'):
        diagram=Diagram(kind)
        path=ROOT/'docs/design/figures'/f'{kind}-demo-v07.pdf'
        canvas=Canvas(str(path),pagesize=(BODY_W+20,diagram.height+20))
        diagram.drawOn(canvas,10,10);canvas.showPage();canvas.save()
        with pymupdf.open(path) as preview:
            preview[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(path.with_suffix('.png'))
        for item in (path,path.with_suffix('.png')):
            demo_files[str(item.relative_to(ROOT))]=sha(item)
    import pymupdf
    pdf=pymupdf.open(args.output)
    baseline=ROOT/'dynamic-warehouse-vln-design-v0.6.pdf'
    git=lambda *a:subprocess.run(['git',*a],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    report=dict(version='v0.7',date='2026-11-27',source_sha256=sha(args.source),pdf_sha256=sha(args.output),
                baseline_sha256=sha(baseline),baseline_pages=len(pymupdf.open(baseline)),pages=len(pdf),
                git_head=git('rev-parse','HEAD'),working_tree_dirty=bool(git('status','--porcelain')),
                fonts={k:dict(path=str(v),sha256=sha(v)) for k,v in fonts.items()},
                dependencies={k:importlib.metadata.version(k) for k in ('reportlab','Markdown','pymupdf','pdfrw')},
                outline_entries=len(pdf.get_toc()),toc_full_row_links=toc_link_rows,render_checks='pending',
                architecture_files=architecture_files,training_files=training_files,original_figure_source=dict(file='dynamic-warehouse-vln-design-v0.6.pdf',page=4,clip=[76,269,580,716],mode='original vector artwork with image overlays and multiview input',controller_input='selected trajectory only'),structure='four-part',diagrams=re.findall(r'<!-- diagram:(\w+) -->',text),
                builder_sha256=sha(__file__),
                demo_files=demo_files,demo_builder_sha256=sha(ROOT/'docs/design/demo_figures.py'),
                dataengine_builder_sha256=sha(ROOT/'docs/design/dataengine_figures.py'),
                dataengine_assets={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'docs/design/figures/dataengine-assets').iterdir() if p.is_file()},
                demo_assets={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'docs/design/figures/warehouse-assets').glob('gsplat-*')},
                illustration_assets={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000197'/n for n in ('rgb.png','detections.png')]+[ROOT/'docs/design/figures/warehouse-assets'/n for n in ('depth.png','global-occupancy.png','left.png','right.png')]+[ROOT/'demo/semantic_mapping/original-20261013/semantic_bev.png']},
                annotation_example_sha256=sha(ROOT/'docs/design/r2r-annotation-example.json'),
                displayed_output_source=dict(file=str((ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000197/parsed.json').relative_to(ROOT)),sha256=sha(ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000197/parsed.json'),origin='existing Codex-session Oracle annotation',waypoint_protocol='figure adapted to two illustrative anchors; original five-point annotation unchanged',figure_waypoints=FIGURE_WAYPOINTS,decoder='not run; trajectory overlay illustrative'),
                asset_provenance={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'docs/design/figures/warehouse-assets').glob('*provenance.json')},
                archived_v07_source_sha256=sha(ROOT/'docs/design/archive/v0.7-20261028.md'),
                note='Documentation build; actual Isaac 5.0 camera re-render and saved map visualization. No model calls, training or navigation validation.')
    for i,page in enumerate(pdf):
        if not page.get_text().strip():raise AssertionError(f'Empty page {i+1}')
    (ROOT/'docs/design/v0.7-build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('pages','outline_entries','source_sha256','pdf_sha256')},indent=2))


if __name__=='__main__':main()
