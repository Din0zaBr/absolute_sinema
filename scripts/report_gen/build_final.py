# -*- coding: utf-8 -*-
import sys, os, glob, json
SCR = os.environ['SCR']; REPO = os.environ['REPO']
sys.path.insert(0, SCR)
from build_report import _load, _crop, build
from PIL import Image, ImageDraw

def g(pat, nopair=False):
    fs = glob.glob(os.path.join(REPO, pat))
    if nopair: fs = [f for f in fs if 'pair' not in os.path.basename(f).lower()]
    assert fs, pat; return fs[0]
def defects(jsonpath):
    j = json.load(open(jsonpath, encoding='utf-8'))
    return sorted(j.get('defects', []), key=lambda d: d.get('confidence',0), reverse=True)
def union_bbox(ds):
    xs=[d['bbox_px'] for d in ds]
    l=min(b[0] for b in xs); t=min(b[1] for b in xs)
    r=max(b[0]+b[2] for b in xs); btm=max(b[1]+b[3] for b in xs)
    return [l,t,r-l,btm-t]

CROP=os.path.join(SCR,"gal_crops"); os.makedirs(CROP,exist_ok=True)
CR=os.path.join(SCR,"crops")

def crop_save(ann, bbox, pad, name, mw=1000, q=85):
    im=_load(ann); c=_crop(im,bbox,pad=pad); w,h=c.size
    if w>mw: c=c.resize((mw,int(h*mw/w)),Image.LANCZOS)
    p=os.path.join(CROP,name+".jpg"); c.save(p,quality=q); return p

# ---------- GALLERY (variety; concrete-pothole reveals moved to Было→стало) ----------
GAL = [
 ("pot_022","outputs_pairs/front_22_annotated.jpg","outputs_pairs/022__*pair.json",
  "dominant",0.6,"Выбоина с оголённым щебнем; контур точно по краю.",[("яма","ok"),("0.83","")]),
 ("JO7VcI","outputs_demo/JO7VcIs*_annotated.jpg","outputs_demo/JO7VcIs*.json",
  "union",0.12,"Яма и сеть трещин в одном кадре — несколько классов сразу.",[("яма + трещины","ok")]),
 ("pot_005","outputs_pairs/front_5_annotated.jpg","outputs_pairs/005__*pair.json",
  "dominant",0.55,"Выбоина на асфальте — уверенная детекция.",[("яма","ok"),("0.86","")]),
 ("pot_044","outputs_pairs/front_44_annotated.jpg","outputs_pairs/044__*pair.json",
  "dominant",0.5,"Выбоина в центре проезда, чёткая маска.",[("яма","ok")]),
 ("hwfU5","outputs_demo/hwfU5q*_annotated.jpg","outputs_demo/hwfU5q*.json",
  "dominant_class:longitudinal_crack",1.0,"Продольная трещина: тонкая структура в полном разрешении.",[("трещина","warn")]),
]
gallery_items=[]
for gid,anng,jsg,mode,pad,cap,badges in GAL:
    ann=g(anng); ds=defects(g(jsg))
    if mode=="union": bbox=union_bbox(ds)
    elif mode.startswith("dominant_class:"):
        cls=mode.split(":")[1]; cand=[d for d in ds if d.get('class')==cls] or ds; bbox=cand[0]['bbox_px']
    else: bbox=ds[0]['bbox_px']
    p=crop_save(ann,bbox,pad,gid)
    gallery_items.append(dict(src=p,caption=cap,badges=[dict(t=t,cls=cl) for t,cl in badges]))

# ---------- BEFORE / AFTER (оригинал -> разметка), same bbox both sides ----------
BA=[
 ("6dyN","6dyN*.jpg","outputs_demo/6dyN*_ensemble_annotated.jpg",[970,238,805,579],0.3,
  "Яма засыпана светлым бетоном — на дороге её легко не заметить. Система выделяет её (второй проход, уверенность 0.81)."),
 ("Q5pslz","Q5pslz*.jpg","outputs_demo/Q5pslz*_annotated.jpg",[368,1192,277,301],0.5,
  "Заделанная выбоина: система обводит контур точно по краю (0.89)."),
]
ba_items=[]
for bid,rawg,anng,bbox,pad,cap in BA:
    raw=g(rawg); ann=g(anng,nopair=True)
    pb=crop_save(raw,bbox,pad,bid+"_raw"); pa=crop_save(ann,bbox,pad,bid+"_ann")
    ba_items.append(dict(before=dict(src=pb),after=dict(src=pa),caption=cap))

# ---------- PAIRS: full frame (marked) + zoom ----------
PAIR_BB={
 "018":{"front":[2021.1,751.0,191.7,98.3],"back":[1948.2,973.9,239.2,131.6]},
 "006":{"front":[2049.3,1170.9,333.8,204.1],"back":[1733.1,746.8,280.3,122.4]},
 "012":{"front":[1884.8,827.1,168.6,166.0],"back":[1631.0,1165.2,176.1,232.4]},
}
def full_marked(raw, bbox, out, mw=940, color=(255,59,48)):
    im=_load(raw).copy(); d=ImageDraw.Draw(im)
    x,y,w,h=bbox; pad=0.22
    box=[x-w*pad, y-h*pad, x+w*(1+pad), y+h*(1+pad)]
    d.rectangle(box, outline=(255,255,255), width=26)   # белый ореол
    d.rectangle(box, outline=color, width=13)           # красная рамка
    ww,hh=im.size
    if ww>mw: im=im.resize((mw,int(hh*mw/ww)),Image.LANCZOS)
    im.save(out,quality=84); return out
for pid,vb in PAIR_BB.items():
    for view in ("front","back"):
        raw=os.path.join(REPO,"datasets","local_pairs",pid,view+".jpg")
        full_marked(raw, vb[view], os.path.join(CROP,f"{pid}_{view}_full.jpg"))

def PV(pid,view,conf,ecc,sol):
    return dict(full=os.path.join(CROP,f"{pid}_{view}_full.jpg"),
                zoom=os.path.join(CR,f"{pid}_{view}.jpg"),conf=conf,ecc=ecc,sol=sol)
PAIRS=[
 dict(label="A · чистая пара",
      front=PV("018","front",0.76,0.92,0.98), back=PV("018","back",0.70,0.86,0.99),
      agree_note="На полном кадре по окружению видно, что это один и тот же участок с двух сторон. Форма согласуется (заполненность 0.98 и 0.99). Размер в см не выдаётся — нет эталона в кадре."),
 dict(label="B · тот же рисунок разрушения",
      front=PV("006","front",0.29,0.90,0.98), back=PV("006","back",0.77,0.94,0.96),
      agree_note="Одна и та же крошащаяся яма у припаркованной машины, снята с двух сторон; даже при низкой уверенности спереди (0.29) контур точный."),
 dict(label="C · выбоина вдоль трещины",
      front=PV("012","front",0.78,0.84,0.95), back=PV("012","back",0.67,0.85,0.88),
      agree_note="Выбоина вдоль трещины опознана с обеих сторон. Без эталона в кадре — только контур, без сантиметров."),
]

# ---------- DIFFPAIRS: общий кадр, РАЗНЫЕ реальные ямы (не сопоставлены как одна) ----------
DIFF_BB={
 "027":{"front":[1695,861,192,108],"back":[1794,630,172,98]},
 "029":{"front":[1864,1237,278,152],"back":[1729,1205,602,259]},
}
for pid,vb in DIFF_BB.items():
    for view in ("front","back"):
        raw=os.path.join(REPO,"datasets","local_pairs",pid,view+".jpg")
        full_marked(raw, vb[view], os.path.join(CROP,f"{pid}_{view}_d4.jpg"))
def D4(pid,view,conf):
    return dict(full=os.path.join(CROP,f"{pid}_{view}_d4.jpg"),cls="яма",conf=conf)
DIFFPAIRS=[
 dict(label="Пример 1 · разные ямы",
      front=D4("027","front",0.32), back=D4("027","back",0.45),
      caption="Спереди система выделила одну яму, сзади — другую (по общему кадру видно, что это разные места). Обе настоящие, но как один объект не сопоставлены."),
 dict(label="Пример 2 · разные ямы",
      front=D4("029","front",0.79), back=D4("029","back",0.84),
      caption="И здесь отмечены разные ямы участка, а не одна и та же. Ложные боксы на номере и колесе в детекцию не вошли."),
]

# ---------- MONTAGE: находит по всему массиву (breadth) ----------
def mont(src,conf):
    mont_items.append(dict(src=src,caption="",badges=[dict(t="яма",cls="ok"),dict(t=conf,cls="")]))
mont_items=[]
Rout=os.path.join(SCR,"regen_out")
for name,fn,bbox,pad,conf in [
    ("m015","c015_front_annotated.jpg",[1688,1015,405,151],0.5,"0.81"),
    ("m028","c028_front_annotated.jpg",[1676,742,309,84],0.9,"0.79")]:
    mont(crop_save(os.path.join(Rout,fn),bbox,pad,name,mw=760),conf)
mont(os.path.join(CR,"018_front.jpg"),"0.76")   # чистая яма, сцена 018
for name,conf in [("Q5pslz","0.89"),("6dyN_ens","0.81"),("pot_022","0.83"),("pot_005","0.86"),("pot_044","0.79")]:
    mont(os.path.join(CROP,name+".jpg"),conf)

# ---------- MISSED: нашла с одного ракурса, пропустила с другого ----------
try:
    from PIL import ImageFont
    _FONT=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 62)
except Exception:
    _FONT=None
def _dashed_rect(d, box, color, width=13, dash=48, gap=32):
    l,t,r,b=[int(v) for v in box]; x=l
    while x<r:
        d.line([x,t,min(x+dash,r),t],fill=color,width=width)
        d.line([x,b,min(x+dash,r),b],fill=color,width=width); x+=dash+gap
    y=t
    while y<b:
        d.line([l,y,l,min(y+dash,b)],fill=color,width=width)
        d.line([r,y,r,min(y+dash,b)],fill=color,width=width); y+=dash+gap
def full_dashed(raw,bbox,out,label="пропущено",mw=940,color=(245,166,35)):
    im=_load(raw).copy(); d=ImageDraw.Draw(im)
    x,y,w,h=bbox; box=[x,y,x+w,y+h]
    _dashed_rect(d,box,color)
    if _FONT is not None:
        tx,ty=int(box[0]),int(box[1])-80
        tb=d.textbbox((tx,ty),label,font=_FONT)
        d.rectangle([tb[0]-10,tb[1]-6,tb[2]+10,tb[3]+8],fill=color)
        d.text((tx,ty),label,fill=(20,20,20),font=_FONT)
    ww,hh=im.size
    if ww>mw: im=im.resize((mw,int(hh*mw/ww)),Image.LANCZOS)
    im.save(out,quality=84); return out
_MB={"001":([1733,1077,235,152],[1930,1000,480,300]),
     "007":([1518,776,300,270],[1720,620,560,440])}
for pid,(fb,mb) in _MB.items():
    full_marked(os.path.join(REPO,"datasets","local_pairs",pid,"back.jpg"),fb,
                os.path.join(CROP,f"{pid}_found.jpg"))
    full_dashed(os.path.join(REPO,"datasets","local_pairs",pid,"front.jpg"),mb,
                os.path.join(CROP,f"{pid}_missed.jpg"))
MISSED=[
 dict(label="Пример 1 · нашла на ближнем — пропустила на дальнем",
      found=dict(full=os.path.join(CROP,"001_found.jpg"),view="сзади · ближе",conf=0.65),
      missed=dict(full=os.path.join(CROP,"001_missed.jpg"),view="спереди · дальше"),
      caption="Ту же яму система уверенно выделила на ближнем кадре (0.65) и не увидела на дальнем — там она мельче и под острым углом. Кадр вблизи убирает пропуск."),
 dict(label="Пример 2 · тот же эффект",
      found=dict(full=os.path.join(CROP,"007_found.jpg"),view="сзади · ближе",conf=0.83),
      missed=dict(full=os.path.join(CROP,"007_missed.jpg"),view="спереди · дальше"),
      caption="И здесь дефект найден на близком ракурсе (0.83), но пропущен на дальнем. Пропуски приходятся на кадры, где яма далеко."),
]

NOTES=[
 dict(icon="📏",t="Сантиметры — только с эталоном",
   d="Размер в см считается лишь когда в кадре есть объект известного размера (люк ГОСТ 3634, разметка, линейка). На этих уличных фото эталона нет — показаны контуры, но не выдуманные сантиметры."),
 dict(icon="📐",t="Глубину одно фото не даёт",
   d="Глубина ямы в см по одному снимку физически не восстановима. Выдаётся только «мелкая / средняя / глубокая»; ГОСТ-глубина — из серии кадров."),
 dict(icon="🔗",t="Пара — когда яма в кадре одна",
   d="Два ракурса связываются, если в каждом кадре одна явная яма. Если ям несколько, система отказывается сопоставлять — лучше не связать, чем связать неверно."),
 dict(icon="⚠️",t="Ложные срабатывания бывают",
   d="На полном кадре модель иногда метит колесо, тень или заделку как яму. Здесь такие боксы отсечены; системно их убирает дообучение на местных фото."),
]

MANIFEST=dict(
 title="Дорожные дефекты глазами модели",
 subtitle="Реальные снимки, реальные контуры. В центре — одна яма, снятая с двух сторон: модель находит её в обоих ракурсах и связывает как один объект.",
 stats=[dict(v="48×2",l="пары «яма с двух сторон»"),
        dict(v="5",l="классов дефектов по ГОСТ"),
        dict(v="~0.2 с",l="на фото, CPU"),
        dict(v="0",l="выдуманных см без эталона")],
 sections=[
   dict(n="1",h="Что модель видит",type="gallery",
        lede="Найденные дефекты обведены прямо на фото. Характерные примеры по типам.",items=gallery_items),
   dict(n="2",h="Было → стало",type="beforeafter",
        lede="Слева — фото, как его видит инспектор. Справа — что находит система за доли секунды.",items=ba_items),
   dict(n="3",h="Одна яма — два ракурса",type="pairs",
        lede="Каждую яму снимают спереди и сзади. Модель находит и обводит доминирующую яму в каждом кадре и связывает их как один объект. Форма ямы в обоих ракурсах согласуется.",items=PAIRS),
   dict(n="4",h="Не всегда та же яма — но находит настоящие",type="diffpairs",
        lede="Иногда систему не удаётся связать одну яму в двух ракурсах — но в каждом кадре она находит реальные дефекты, просто разные. Показываем и это честно.",items=DIFFPAIRS),
   dict(n="5",h="Иногда видит только с одного ракурса",type="missedpairs",
        lede="Бывает и так: одну и ту же яму система выделяет с одного ракурса и пропускает с другого.",
        tip="<b>Снимайте вблизи — точность заметно выше.</b> Пропуски почти всегда приходятся на кадр, где яма далеко или под острым углом. Чем крупнее яма в кадре (камера ближе и чуть сверху), тем больше пикселей — и детекция надёжнее.",
        items=MISSED),
   dict(n="6",h="Честные границы",type="notes",
        lede="Где модель осознанно не угадывает — и где ещё ошибается.",items=NOTES),
 ],
 footer="Road Defect Engine · детектор YOLO (RDD2022) + сегментация MobileSAM · все контуры получены на этих фото офлайн. Кадры прошли визуальную проверку; отдельные полнокадровые оверлеи содержат ложные боксы — см. «честные границы».",
)
build(MANIFEST, os.path.join(REPO,"docs","detection_report.html"))
