"""Build the Hungarian report and figures from saved results. NEVER trains a model."""
import argparse
import csv
import html
import importlib.metadata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from common.evaluation import iou, operating_point
from common.utils import ROOT, fingerprint, load_config, read_json, write_json

NAMES = {"yolo": "YOLOv8s", "faster_rcnn": "Faster R-CNN"}
HU = {"Impacted": "Elő nem tört fog", "Caries": "Szuvasodás", "Periapical Lesion": "Gyökércsúcs körüli elváltozás", "Deep Caries": "Mély szuvasodás"}
COLORS = {"yolo": "#2374ab", "faster_rcnn": "#db6639"}


def pct(value):
    return f"{100*value:.2f}%".replace(".", ",") if value is not None else "N/A"


def number(value, decimals=2):
    return f"{value:.{decimals}f}".replace(".", ",")


def duration(seconds):
    seconds = round(seconds)
    return f"{seconds//3600} óra {(seconds%3600)//60} perc {seconds%60} mp"


def rows_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def matched_predictions(gt, predictions, threshold):
    truth = defaultdict(list)
    for a in gt["annotations"]:
        truth[(a["image_id"], a["category_id"])].append(a)
    used, found = set(), []
    for p in sorted(predictions, key=lambda p: p["score"], reverse=True):
        if p["score"] < threshold:
            continue
        key = (p["image_id"], p["category_id"])
        choices = [(iou(p["bbox"], a["bbox"]), a["id"]) for a in truth[key] if a["id"] not in used]
        overlap, ident = max(choices, default=(0, -1))
        good = overlap >= 0.5
        if good:
            used.add(ident)
        found.append({**p, "match": "TP" if good else "FP"})
    return found, [a for a in gt["annotations"] if a["id"] not in used]


def check_inputs(cfg):
    root = Path(cfg["prepared_dir"])
    manifest = read_json(root / "manifest.json")
    reports = {m: read_json(ROOT / "runs" / m / "test_evaluation.json") for m in NAMES}
    datasets = {s: read_json(root / f"{s}.json") for s in ("train", "val", "test")}
    for method, report in reports.items():
        meta = report["metadata"]
        assert meta["split"] == "test", "Test evaluations required"
        assert meta["method"] == method
        assert meta["manifest_sha256"] == fingerprint(root / "manifest.json")
        assert meta["ground_truth_sha256"] == fingerprint(root / "test.json")
        weights = ROOT / "runs" / method / ("weights/best.pt" if method == "yolo" else "best.pth")
        assert meta["weights_sha256"] == fingerprint(weights), "Weights changed since evaluation"
        recalculated = operating_point(datasets["test"],report["predictions"],meta["display_threshold"])
        assert all(recalculated[k] == report["metrics"][k] for k in recalculated)
    for key in ("device", "image_size", "score_floor", "display_threshold", "max_detections", "nms_iou", "images"):
        assert reports["yolo"]["metadata"][key] == reports["faster_rcnn"]["metadata"][key], f"Different {key}"
    ids = [set(manifest["splits"][s]) for s in ("train", "val", "test")]
    assert all(not a & b for i,a in enumerate(ids) for b in ids[i+1:])
    assert all(ids[i] == {im["id"] for im in datasets[s]["images"]} for i,s in enumerate(("train","val","test")))
    return manifest, datasets, reports


def generate_analysis(manifest, datasets, reports, folder):
    folder.mkdir(parents=True, exist_ok=True)
    errors = []
    threshold = reports["yolo"]["metadata"]["display_threshold"]
    for method, report in reports.items():
        for c in manifest["categories"]:
            gt = {**datasets["test"], "annotations": [a for a in datasets["test"]["annotations"] if a["category_id"] == c["id"]]}
            pred = [p for p in report["predictions"] if p["category_id"] == c["id"]]
            metric = operating_point(gt,pred,threshold)
            errors.append({"method": method, "class": c["name"], "annotations": len(gt["annotations"]), **metric})
        for key in ("tp", "fp", "fn"):
            assert sum(r[key] for r in errors if r["method"] == method) == report["metrics"][key]
    with (folder / "per_class_errors.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=list(errors[0]))
        writer.writeheader()
        writer.writerows(errors)
    duplicates = {}
    for split, data in datasets.items():
        counter = Counter((a["image_id"], a["category_id"], tuple(a["bbox"])) for a in data["annotations"])
        duplicates[split] = sum(n-1 for n in counter.values() if n > 1)
    environment = {p: importlib.metadata.version(p) for p in ("torch","torchvision","ultralytics","numpy","opencv-python","Pillow","pycocotools","matplotlib","reportlab")}
    audit = {"manifest_sha256": reports["yolo"]["metadata"]["manifest_sha256"], "image_counts": {s:len(d["images"]) for s,d in datasets.items()}, "annotation_counts": {s:len(d["annotations"]) for s,d in datasets.items()}, "duplicate_annotations": duplicates, "disjoint_splits": True, "saved_metric_counts_verified": True, "weights_hashes_verified": True, "environment_at_report_generation": environment, "per_class_errors": errors}
    write_json(folder / "analysis.json",audit)
    return errors, audit


def plots(manifest, reports, output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.axisbelow":True})
    names = [c["name"] for c in manifest["categories"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.1,3.5),layout="constrained")
    for i,(method,report) in enumerate(reports.items()):
        vals = [100*report["metrics"]["per_class_map50_95"][name] for name in names]
        bars = ax.bar(x+(i-.5)*.36,vals,width=.36,label=NAMES[method],color=COLORS[method])
        ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    ax.set_xticks(x,["Elő nem tört fog","Szuvasodás","Gyökércsúcs körüli\nelváltozás","Mély szuvasodás"])
    ax.set_ylabel("AP@0.50:0.95 (%)")
    ax.set_ylim(0,70)
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.2)
    fig.savefig(output / "class_ap.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(8.1,3.2),layout="constrained")
    for ax,key,title,unit in [(axes[0],"latency_ms_per_image","Feldolgozási idő (batch=1)","ms/kép"),(axes[1],"training_seconds","Teljes tanítási idő","perc")]:
        vals = [r["metadata"][key] / (60 if key == "training_seconds" else 1) for r in reports.values()]
        bars=ax.bar(list(NAMES.values()),vals,color=list(COLORS.values()),width=.55)
        ax.bar_label(bars,fmt="%.1f",padding=3)
        ax.set_title(title,fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0,max(vals)*1.2)
        ax.grid(axis="y",alpha=.2)
    fig.savefig(output / "runtime.png",dpi=180)
    plt.close(fig)
    histories = {"yolo":rows_csv(ROOT / "runs/yolo/results.csv"),"faster_rcnn":rows_csv(ROOT / "runs/faster_rcnn/history.csv")}
    best = {}
    fig,axes=plt.subplots(1,2,figsize=(8.1,3.3),layout="constrained")
    for ax,(method,rows) in zip(axes,histories.items()):
        key = "metrics/mAP50-95(B)" if method == "yolo" else "val_map50_95"
        ep=[int(r["epoch"]) for r in rows]
        vals=[100*float(r[key]) for r in rows]
        highest=max(rows,key=lambda r:float(r[key]))
        best[method]={"epoch_of_highest_logged_val_map":int(highest["epoch"]),"highest_logged_val_map":float(highest[key]),"epochs_completed":len(rows)}
        ax.plot(ep,vals,color=COLORS[method],linewidth=2)
        ax.axvline(int(highest["epoch"]),color="#555555",linestyle="--",alpha=.65,label=f"Legnagyobb naplózott AP: {highest['epoch']}. epoch")
        ax.set_title(NAMES[method])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validációs mAP@0.50:0.95 (%)")
        ax.legend(frameon=False,fontsize=8,loc="lower right")
        ax.grid(alpha=.2)
    fig.savefig(output / "validation_curves.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.1,3.0),layout="constrained")
    for ax,(method,rows) in zip(axes,histories.items()):
        keys=["train/box_loss","train/cls_loss","train/dfl_loss"] if method == "yolo" else ["train_loss"]
        for key in keys:
            ax.plot([int(r["epoch"]) for r in rows],[float(r[key]) for r in rows],label=key)
        ax.set_title(NAMES[method])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Saját tanítási veszteség")
        ax.legend(frameon=False,fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(output / "loss_curves.png",dpi=180)
    plt.close(fig)
    return best


def example_figures(cfg, datasets, reports, output):
    cats = datasets["test"]["categories"]
    result=[]
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"),20)
    palette=["#e63946","#ff9f1c","#2a9d8f","#4361ee"]
    def panel(source, annotations, reference=False):
        img=source.copy()
        draw=ImageDraw.Draw(img)
        reserved=[]
        for a in annotations:
            x,y,w,h=a["bbox"]
            color=palette[a["category_id"]]
            draw.rectangle((x,y,x+w,y+h),outline=color,width=3)
        for a in annotations:
            x,y,w,h=a["bbox"]
            name=cats[a["category_id"]]["name"]
            label=name if reference else f"{a['match']} {name} {a['score']:.2f}"
            label_width=draw.textlength(label,font=font)
            px=min(x,max(0,img.width-label_width-4))
            py=max(0,y-28)
            pos=(px,py)
            bounds=draw.textbbox(pos,label,font=font)
            while any(bounds[0] < b[2]+3 and bounds[2]+3 > b[0] and bounds[1] < b[3]+3 and bounds[3]+3 > b[1] for b in reserved):
                py=py-26 if py >= 26 else y+h+8+26*len(reserved)
                pos=(px,py)
                bounds=draw.textbbox(pos,label,font=font)
            reserved.append(bounds)
            if abs(py-y)>40:
                draw.line((px+8,bounds[3],x+8,y),fill="white",width=1)
            draw.rectangle(bounds,fill="#101828")
            draw.text(pos,label,fill="white",font=font)
        return img
    for im in datasets["test"]["images"][:3]:
        path=Path(cfg["prepared_dir"]) / "images/test" / im["file_name"]
        with Image.open(path) as image:
            source=image.convert("RGB")
        bounds=ImageChops.difference(source,Image.new("RGB",source.size,(114,114,114))).getbbox()
        l,t,r,b=bounds
        bounds=(max(0,l-8),max(0,t-24),min(source.width,r+8),min(source.height,b+8))
        truth=[a for a in datasets["test"]["annotations"] if a["image_id"]==im["id"]]
        fig,axes=plt.subplots(3,1,figsize=(8.1,9.2),layout="constrained")
        axes[0].imshow(panel(source,truth,True).crop(bounds))
        axes[0].set_title(f"Referencia - {len(truth)} annotáció",loc="left",fontsize=11)
        stats={}
        for ax,(method,report) in zip(axes[1:],reports.items()):
            pred=[p for p in report["predictions"] if p["image_id"]==im["id"]]
            subset={**datasets["test"],"annotations":truth}
            selected,missed=matched_predictions(subset,pred,report["metadata"]["display_threshold"])
            metrics=operating_point(subset,pred,report["metadata"]["display_threshold"])
            stats[method]=metrics
            ax.imshow(panel(source,selected).crop(bounds))
            ax.set_title(f"{NAMES[method]} - TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}",loc="left",fontsize=11)
        for ax in axes:
            ax.axis("off")
        figure=output / f"example_{im['id']}.png"
        fig.savefig(figure,dpi=180)
        plt.close(fig)
        result.append({"id":im["id"],"annotations":len(truth),"stats":stats,"figure":figure})
    return result


class Report:
    def __init__(self, pdf_path, markdown_path):
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate
        from reportlab.lib.pagesizes import A4
        fonts=Path(matplotlib.get_data_path()) / "fonts/ttf"
        pdfmetrics.registerFont(TTFont("DV",str(fonts / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DV-Bold",str(fonts / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily("DV",normal="DV",bold="DV-Bold",italic="DV",boldItalic="DV-Bold")
        self.styles=getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="HU",fontName="DV",fontSize=9.5,leading=14,spaceAfter=8,textColor=colors.HexColor("#243247")))
        self.styles.add(ParagraphStyle(name="HUTitle",fontName="DV-Bold",fontSize=23,leading=30,spaceAfter=18,textColor=colors.HexColor("#12304a")))
        self.styles.add(ParagraphStyle(name="HUHeading",fontName="DV-Bold",fontSize=16,leading=22,spaceAfter=12,keepWithNext=True,textColor=colors.HexColor("#12304a")))
        self.styles.add(ParagraphStyle(name="HUSmall",fontName="DV",fontSize=8,leading=11,spaceAfter=6))
        self.styles.add(ParagraphStyle(name="HUCell",fontName="DV",fontSize=8,leading=11))
        self.pdf_path,self.md_path=pdf_path,markdown_path
        self.width=A4[0]-100
        self.doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=46,bottomMargin=45,title="Fogászati röntgenképek elemzése - YOLO és Faster R-CNN",author="Egyetemi projekt")
        self.story,self.md=[],[]

    def heading(self,text,title=False):
        from reportlab.platypus import Paragraph
        self.story.append(Paragraph(html.escape(text),self.styles["HUTitle" if title else "HUHeading"]))
        self.md.extend([("# " if title else "## ")+text,""])

    def p(self,text,small=False):
        from reportlab.platypus import Paragraph
        self.story.append(Paragraph(html.escape(text),self.styles["HUSmall" if small else "HU"]))
        self.md.extend([text,""])

    def table(self,headers,rows,widths=None):
        from reportlab.platypus import Table,TableStyle,Paragraph,Spacer
        from reportlab.lib import colors
        data=[[Paragraph(html.escape(str(v)),self.styles["HUCell"]) for v in row] for row in [headers,*rows]]
        widths=[self.width*w/sum(widths) for w in widths] if widths else [self.width/len(headers)]*len(headers)
        table=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7eef5")),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f8fb")]),("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#a5b7c7")),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        self.story.extend([table,Spacer(1,10)])
        self.md.extend(["| "+" | ".join(map(str,headers))+" |","| "+" | ".join(["---"]*len(headers))+" |"])
        self.md.extend("| "+" | ".join(map(str,row))+" |" for row in rows)
        self.md.append("")

    def figure(self,path,caption,maxheight=240):
        from reportlab.platypus import Image as RLImage,Paragraph,KeepTogether,Spacer
        with Image.open(path) as img:
            w,h=img.size
        scale=min(self.width/w,maxheight/h)
        self.story.append(KeepTogether([RLImage(str(path),width=w*scale,height=h*scale),Spacer(1,5),Paragraph(html.escape(caption),self.styles["HUSmall"])]))
        import os
        relative=Path(os.path.relpath(path,self.md_path.parent)).as_posix()
        self.md.extend([f"![{caption}]({relative})","",caption,""])

    def page(self):
        from reportlab.platypus import PageBreak
        self.story.append(PageBreak())
        self.md.extend(["---",""])

    def save(self):
        self.pdf_path.parent.mkdir(parents=True,exist_ok=True)
        self.md_path.parent.mkdir(parents=True,exist_ok=True)
        def footer(canvas,doc):
            canvas.setFont("DV",8)
            canvas.setFillColorRGB(.35,.4,.47)
            canvas.drawString(50,25,"DENTEX | YOLOv8s és Faster R-CNN | Egyetemi projekt")
            canvas.drawRightString(545,25,str(doc.page))
        self.doc.build(self.story,onFirstPage=footer,onLaterPages=footer)
        self.md_path.write_text("\n".join(self.md),encoding="utf-8")


def author_report(cfg,meta,manifest,datasets,reports,errors,audit,best,examples,figures):
    report=Report(ROOT / "output/pdf/fogaszati_ai_beszamolo.pdf",ROOT / "docs/beszamolo.md")
    y,f=reports["yolo"],reports["faster_rcnn"]
    report.heading("Fogászati röntgenképek elemzése mesterséges intelligenciával",title=True)
    report.p("YOLOv8s és Faster R-CNN összehasonlítása a DENTEX adathalmazon")
    report.table(["Adat","Érték"],[["Szerző",meta["author"]],["Egyetem / szak",meta["institution"]],["Tantárgy",meta["course"]],["Kísérlet dátuma",meta["date"]]],widths=[1,2])
    report.heading("Összefoglaló")
    report.p("A projekt célja négy fogászati eltéréssel megjelölt fog lokalizálása panorámaröntgeneken. Két előtanított objektumdetektort finomhangoltunk ugyanazon a 493 tanítóképen: YOLOv8s és Faster R-CNN ResNet-50 FPN. Modellválasztásra 106 validációs képet, végső összehasonlításra további 106, a tanításból kihagyott tesztképet használtunk.")
    report.p(f"A közös COCO kiértékelővel mért teszt mAP@0.50:0.95 a YOLO esetén {pct(y['metrics']['map50_95'])}, a Faster R-CNN esetén {pct(f['metrics']['map50_95'])}. A Faster R-CNN ebben a kísérletben jobb detektálási eredményt adott, míg a YOLO {f['metadata']['latency_ms_per_image']/y['metadata']['latency_ms_per_image']:.2f}-szer gyorsabban dolgozott fel egy képet. A teljes tanítási idő rendre {duration(y['metadata']['training_seconds'])} és {duration(f['metadata']['training_seconds'])} volt.")
    report.p("A kész rendszer képenként kereteket, osztálycímkéket és pontszámokat ad. Nem állít elő szöveges orvosi leletet, nem végez fogszámozást, és az eredmény nem tekinthető klinikai diagnózisnak. Az mAP detektálási mérőszám, nem egyszerű osztályozási találati arány.")
    report.p("A beszámoló a már befejezett tanítások naplóiból és mentett tesztpredikcióiból készült. Elkészítéséhez nem történt új tanítás.")
    report.page()
    report.heading("1. Adathalmaz és feladat")
    report.p("A DENTEX hierarchikus annotációkat tartalmaz: kvadráns, fogazonosítás és diagnózis. A feladathoz a training_data/quadrant-enumeration-disease részhalmazt használtuk, 705 képpel és 3529 annotációval. A quadrant, quadrant_enumeration és unlabelled almappák nem kerültek a diagnózisdetektorok tanításába. Forrás: a DENTEX készítőinek leírása és a helyi train_quadrant_enumeration_disease.json [1, 2].")
    report.table(["Diagnózis","Tanító","Validáció","Teszt"],[[HU[c["name"]],*[manifest["counts"][s][c["name"]] for s in ("train","val","test")]] for c in manifest["categories"]]+[["Összes annotáció",*[len(datasets[s]["annotations"]) for s in ("train","val","test")]], ["Képek száma",493,106,106]],widths=[2.2,1,1,1])
    report.p("A felosztás 70/15/15 arányú, képszintű, seed=42 véletlenmaggal. A pontos képazonosítókat a manifest.json rögzíti. A három halmaz diszjunkt. A betegazonosítók hiánya miatt ez nem bizonyítottan betegszintű szétválasztás; ugyanazon beteg eltérő felvételeinek esetleges átfedése nem zárható ki.")
    report.p("A tanítóhalmazban a szuvasodás 1546, a gyökércsúcs körüli elváltozás 117 annotációt kapott: az osztályeloszlás erősen egyenlőtlen. A tesztben az utóbbi osztályhoz mindössze 16 annotáció tartozik, ezért ennél az osztálynál különösen bizonytalan az eredmény általánosíthatósága.")
    report.p("Előfeldolgozás: minden képet RGB-re alakítunk, aránytartóan átméretezünk, majd 1024x1024-es, 114-es szürke kitöltésű vászonra helyezünk. A keretek ugyanazzal a méretezéssel és eltolással alakulnak át. Mindkét modell ugyanazokat az előkészített PNG fájlokat kapja. A YOLO normalizált középpont-szélesség-magasság formátumot, a Faster R-CNN xyxy kereteket használ.")
    report.p(f"Adatminőségi megjegyzés: az egy kép/osztály/keret kombináció szerint ismétlődő annotációk száma train/val/test sorrendben {audit['duplicate_annotations']['train']}/{audit['duplicate_annotations']['val']}/{audit['duplicate_annotations']['test']}. A validációban lévő egy ismétlődő jelölést a YOLO betöltője összevonja (513 jelölés), a közös JSON 514-et tartalmaz. A végső tesztben nincs ilyen eltérés. Az eredeti tanításokat és címkéket utólag nem módosítottuk.")
    report.page()
    report.heading("2. Módszerek és kísérleti beállítások")
    report.p("A YOLOv8s egyfázisú detektor: a hálózat egyetlen feldolgozási láncban becsül kereteket és osztályokat. A Faster R-CNN kétfázisú: az RPN régiójavaslatokat készít, majd a régiónkénti fej osztályoz és pontosítja a kereteket. A ResNet-50 FPN több léptékű jellemzőtérképeket biztosít. Mindkét modell COCO-n előtanított súlyokból indul [3, 4].")
    report.table(["Beállítás","YOLOv8s","Faster R-CNN"],[["Bemenet","1024x1024","1024x1024"],["Epoch","50","50"],["Batch","4","2"],["Optimalizáló","SGD, momentum 0.937","SGD, momentum 0.9"],["Kezdeti learning rate","0.001","0.005"],["Ütemezés","Ultralytics alapütemezés, warmup","StepLR: 16 epochonként x0.1"],["Előtanítás","COCO","COCO"],["Augmentáció","A megadott geometriai/színaugmentációk kikapcsolva","Nincs"],["Kevert pontosság","AMP bekapcsolva","FP32"],["Modellválasztás","Ultralytics validációs fitness","Közös validációs mAP@0.50:0.95"]],widths=[1.2,1.6,1.6])
    report.p(f"Hardver: {meta['gpu']}, 6 GB VRAM; Windows, Python 3.10.4; PyTorch {audit['environment_at_report_generation']['torch']}, Torchvision {audit['environment_at_report_generation']['torchvision']}, Ultralytics {audit['environment_at_report_generation']['ultralytics']}. A tanítások seed=42 beállítással és egymás után futottak. A teljes, aktuálisan rögzített csomaglista a reports/environment.txt fájlban található; ez a beszámoló készítésekor mentett környezet.")
    report.p("A két eljárás tanítási ütemezése, batchmérete és numerikus pontossága eltér. Azonos epochsám ezért nem azonos számítási költség vagy teljesen kontrollált architektúra-összehasonlítás. Az eredmények a megadott konfigurációkra vonatkoznak; nem bizonyítják valamelyik módszer általános fölényét.")
    report.p("Kiértékelés: osztályhelyes keretillesztés; COCO AP 0.50-től 0.95-ig, 0.05 lépésű IoU-küszöbökkel. A pontszám alsó határa AP-hoz 0.001, az NMS IoU-küszöbe 0.5, a maximális detekciószám 100. Precision/recall/F1: rögzített 0.25 pontszámküszöb és IoU legalább 0.5, képenként és osztályonként egy-egy párosítással. A tesztet nem használtuk küszöbhangolásra.")
    report.page()
    report.heading("3. A közös teszt eredményei")
    report.table(["Mérőszám","YOLOv8s","Faster R-CNN"],[[name,pct(y["metrics"][key]),pct(f["metrics"][key])] for name,key in [("mAP@0.50:0.95","map50_95"),("mAP@0.50","map50"),("mAP@0.75","map75"),("Precision (0.25)","precision"),("Recall (0.25)","recall"),("F1 (0.25)","f1")]]+[["Átlagos késleltetés",number(y["metadata"]["latency_ms_per_image"])+" ms/kép",number(f["metadata"]["latency_ms_per_image"])+" ms/kép"],["Teljes tanítási idő",duration(y["metadata"]["training_seconds"]),duration(f["metadata"]["training_seconds"])],["Paraméterszám",number(y["metadata"]["parameters_millions"])+" millió",number(f["metadata"]["parameters_millions"])+" millió"]],widths=[1.5,1,1])
    report.figure(figures / "runtime.png","1. ábra. Feldolgozási és teljes tanítási idő a tényleges naplókból.",maxheight=205)
    report.p(f"A Faster R-CNN mAP@0.50:0.95 előnye {number(100*(f['metrics']['map50_95']-y['metrics']['map50_95']))} százalékpont. A recall különbsége {number(100*(f['metrics']['recall']-y['metrics']['recall']))} százalékpont a rögzített működési ponton. A jobb érzékenység több téves jelöléssel is jár: a Faster R-CNN {f['metrics']['fp']}, a YOLO {y['metrics']['fp']} téves pozitív detekciót adott.")
    report.p("A késleltetés batch=1 mellett, három bemelegítő futás után, GPU-szinkronizálással készült. Tartalmazza az elő-/utófeldolgozást és adatátvitelt, a lemezolvasást nem. Egyetlen mérés átlaga, nem több független futás statisztikai eredménye. A mentett fájlméret nem tiszta modellméret: a Faster R-CNN ellenőrzőpont optimalizálóállapotot is tartalmaz.")
    report.page()
    report.heading("4. Osztályonkénti eredmények és hibák")
    report.figure(figures / "class_ap.png","2. ábra. Osztályonkénti AP@0.50:0.95 a 106 tesztképen.",maxheight=210)
    report.table(["Osztály / modell","TP","FP","FN","Precision","Recall"],[[f"{r['class']} / {'YOLO' if r['method']=='yolo' else 'FRCNN'}",r["tp"],r["fp"],r["fn"],pct(r["precision"]),pct(r["recall"])] for r in errors],widths=[3, .6,.6,.6,1,1])
    report.p("TP: helyesen lokalizált és helyesen osztályozott jelölés. FP: nem párosítható predikció; lehet hibás osztály, elégtelen térbeli átfedés, duplikált detekció vagy annotáció nélküli terület. FN: a rögzített küszöbnél meg nem talált annotáció. Ezek detektálási hibák, önmagukban nem klinikai tévedéskategóriák.")
    report.p("Mindkét modell az elő nem tört fog osztályában érte el a legmagasabb AP-t. A Faster R-CNN mind a négy osztály AP-jában jobb volt ebben a futásban; a mély szuvasodásnál és a gyökércsúcs körüli elváltozásnál nagyobb a különbség. Az eltérések okát e kísérlet nem választja szét: közrejátszhat a hálózat, az optimalizálás és az egyenlőtlen osztályeloszlás is.")
    report.p("A YOLO a gyökércsúcs körüli elváltozásokból a 0.25-ös küszöbnél egyet sem párosított helyesen. Az osztály AP-ja ettől még nem nulla, mert az AP számítása a 0.001-ig megtartott, alacsonyabb pontszámú predikciókat is figyelembe veszi. A fix küszöbnél mért recall és az AP eltérő kérdésre válaszol.")
    report.page()
    report.heading("5. Tanulási görbék és modellválasztás")
    report.figure(figures / "validation_curves.png","3. ábra. A tanítás során naplózott validációs mAP. A YOLO saját, a Faster R-CNN közös kiértékelőt használ; a görbék ezért nem teljesen azonos mérési eljárást ábrázolnak.",maxheight=205)
    report.figure(figures / "loss_curves.png","4. ábra. Saját tanítási veszteségek. A különböző loss-függvények abszolút értékei a két hálózat között nem hasonlíthatók közvetlenül.",maxheight=190)
    report.p(f"Mindkét tanítás befejezte az 50 epochot. A Faster R-CNN legjobb validációs eredménye a {best['faster_rcnn']['epoch_of_highest_logged_val_map']}. epochban {pct(best['faster_rcnn']['highest_logged_val_map'])}; ezt a best.pth őrzi. A YOLO naplózott validációs mAP maximuma a {best['yolo']['epoch_of_highest_logged_val_map']}. epochban található; a teszteléshez az Ultralytics által mentett best.pt került betöltésre.")
    report.p("A Faster R-CNN későbbi epochjai nem javították tovább a validációs maximumot, miközben a tanítási veszteség alacsony maradt. Ez a mintázat összeegyeztethető túltanulással vagy validációs ingadozással, de önmagában nem bizonyítja az okot. A best ellenőrzőpont választása biztosítja, hogy nem szükségszerűen az utolsó epoch modelljét értékeljük.")
    report.p("Új tanítás a projekt lezárásához nem szükséges. A további elemzések és példák a mentett predikciókból állnak elő; a súlyok és a felosztás változatlanok maradnak.")
    for index,example in enumerate(examples,1):
        report.page()
        report.heading(f"6.{index}. Képes példa - tesztazonosító: {example['id']}")
        report.p("A manifest szerinti első három tesztkép egyikét mutatjuk; a példákat nem teljesítmény szerint válogattuk. Felül a referencia, alatta a két modell 0.25 feletti pontszámú detekciói láthatók. A jobb olvashatóságért a szürke paddinget levágtuk; az eredeti predikciókat nem módosítottuk.",small=True)
        report.figure(example["figure"],f"{4+index}. ábra. Azonos kép és azonos működési pont. TP/FP: a megjelenített predikció párosítása; FN: a hiányzó referenciák darabszáma a panel címében.",maxheight=565)
        ys,fs=example["stats"]["yolo"],example["stats"]["faster_rcnn"]
        report.p(f"A {example['annotations']} referenciából a YOLO {ys['tp']}, a Faster R-CNN {fs['tp']} jelölést talált meg. A téves pozitív jelölések száma rendre {ys['fp']} és {fs['fp']}. Ez egy képre vonatkozó leírás; az összesített következtetés a teljes teszthalmazon alapul.",small=True)
    report.page()
    report.heading("7. Következtetések és korlátok")
    report.p("Az elvégzett kísérlet megvalósította a négy diagnózisosztályhoz tartozó fogak lokalizálását két külön megoldással. A Faster R-CNN magasabb mAP-t és recallt adott, a YOLO rövidebb tanítási időt és kisebb képenkénti késleltetést ért el. Ha az adott kísérletben a detektálási eredmény az elsődleges, a Faster R-CNN a kedvezőbb; ha a sebesség és az erőforrásigény fontos, a YOLO előnyös. Egyik modell sem teljesít elég megbízhatóan ahhoz, hogy emberi ellenőrzés nélkül diagnosztikai döntést hozzon.")
    report.p("A következtetések korlátai: egy véletlenmag és egy adatfelosztás; betegszintű elkülönítés nem igazolt; ritka diagnózisok kevés tesztannotációval; eltérő batch, tanítási ütemezés és AMP; egyetlen késleltetésmérés; validációs annotációban egy duplikáció. A végső teszt mindkét módszernél ugyanazt az annotációt és kiértékelőt használta. A pontszámok nincsenek klinikai valószínűségként kalibrálva.")
    report.p("A jövőbeli munka lehet betegszintű és több seedes kísérlet, előre rögzített augmentáció, ritka osztályok célzott kezelése vagy a learning rate validációs hangolása. Ezek fejlesztési lehetőségek, nem elvégzett kísérletek és nem feltételei a jelen projekt befejezésének.")
    report.heading("8. Reprodukálhatóság és bemutatás")
    report.p("A teljes forrás, konfiguráció, modellmetaadat, tanítási napló, tesztpredikció és SHA-256 ellenőrzés megmaradt. A reports/analysis.json rögzíti a súlyok és felosztások ellenőrzését; a reports/per_class_errors.csv az osztályonkénti hibákat. A build_report.py kizárólag mentett eredményeket elemez, nem tanít hálózatot.")
    report.p("Bemutatás új tanítás nélkül: a README.md parancsaival egy képet a predict.py dolgoz fel a best.pt vagy best.pth fájlból. A kimenet jelölt kép és eredeti koordinátás JSON. A beszámoló újragenerálása: python build_report.py. Az előtanított súlyok megléte után az inferencia és a kiértékelés offline is futtatható.")
    report.heading("Hivatkozások és felhasználás")
    for text in ["[1] Hamamci és mtsai. (2023): DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays. https://arxiv.org/abs/2305.19112", "[2] DENTEX hivatalos projekt és adatlicenc (CC BY-NC-SA 4.0). https://github.com/ibrahimethemhamamci/DENTEX ; adat: https://zenodo.org/records/7812323", "[3] Ultralytics YOLO tanítás és validáció. https://docs.ultralytics.com/modes/train/ ; https://docs.ultralytics.com/modes/val/", "[4] Torchvision Faster R-CNN API. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html", "A forrásadatok és a belőlük készült képes példák nem kereskedelmi felhasználásánál a DENTEX adatlicencét, a modellek és könyvtárak esetén azok saját licencfeltételeit kell követni."]:
        report.p(text,small=True)
    report.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="config.json")
    parser.add_argument("--metadata",default="report_metadata.json")
    args=parser.parse_args()
    cfg=load_config(args.config)
    meta=read_json(ROOT / args.metadata)
    manifest,datasets,reports=check_inputs(cfg)
    folder=ROOT / "reports"
    errors,audit=generate_analysis(manifest,datasets,reports,folder)
    figures=folder / "figures"
    best=plots(manifest,reports,figures)
    examples=example_figures(cfg,datasets,reports,figures)
    write_json(folder / "learning_summary.json",best)
    author_report(cfg,meta,manifest,datasets,reports,errors,audit,best,examples,figures)
    environment="\n".join(f"{name}=={version}" for name,version in audit["environment_at_report_generation"].items())
    (folder / "environment.txt").write_text(environment+"\n",encoding="utf-8")
    print("Kész: docs/beszamolo.md; output/pdf/fogaszati_ai_beszamolo.pdf; reports/")
    print("Nem történt új tanítás vagy új inferencia.")


if __name__ == "__main__":
    main()
