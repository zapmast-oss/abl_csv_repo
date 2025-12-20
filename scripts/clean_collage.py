from PIL import Image, ImageDraw, ImageFont
import numpy as np, cv2, os

def clean_collage(src_path, captions, out_path, thr=242):
    img=Image.open(src_path).convert("RGB")
    arr=np.array(img)
    gray=cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    _,th=cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    kernel=np.ones((5,5),np.uint8)
    th2=cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th2, connectivity=8)

    candidates=[]
    H,W=arr.shape[:2]
    for i in range(1, num_labels):
        x,y,w,h,area = stats[i]
        if area < 20000:
            continue
        # avoid full image background
        if w > W*0.98 and h > H*0.98:
            continue
        # white boxes in these screenshots are roughly portrait-like
        # also avoid thin toolbars
        if h < 200 or w < 200:
            continue
        candidates.append((area,i))
    candidates=sorted(candidates, reverse=True)[:4]
    if len(candidates) != 4:
        # fallback: just take top 4 regardless of some filters
        areas=stats[1:,cv2.CC_STAT_AREA]
        top4=np.argsort(areas)[::-1][:4] + 1
        candidates=[(stats[i,cv2.CC_STAT_AREA],i) for i in top4]

    top4=[i for _,i in candidates]
    boxes=[stats[i,:4] for i in top4]
    cent=[centroids[i] for i in top4]
    items=list(zip(boxes, cent))
    items_sorted=sorted(items, key=lambda bc: (bc[1][1], bc[1][0]))

    top_row=sorted(items_sorted[:2], key=lambda bc: bc[1][0])
    bottom_row=sorted(items_sorted[2:], key=lambda bc: bc[1][0])
    ordered=top_row+bottom_row  # tl,tr,bl,br

    crops=[]
    for (x,y,w,h),_ in ordered:
        pad=2
        crops.append(img.crop((x+pad,y+pad,x+w-pad,y+h-pad)))

    # Layout
    margin=28
    gap=28
    caption_h=42
    tile_w, tile_h=crops[0].size
    out_w = margin*2 + tile_w*2 + gap
    out_h = margin*2 + (tile_h+caption_h)*2 + gap

    canvas=Image.new("RGB",(out_w,out_h),"white")
    draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font=ImageFont.load_default()

    positions=[
        (margin, margin),
        (margin+tile_w+gap, margin),
        (margin, margin+tile_h+caption_h+gap),
        (margin+tile_w+gap, margin+tile_h+caption_h+gap),
    ]
    for idx,(x0,y0) in enumerate(positions):
        cap=captions[idx]
        bbox=draw.textbbox((0,0), cap, font=font)
        text_w=bbox[2]-bbox[0]
        tx=x0 + (tile_w-text_w)//2
        draw.text((tx, y0), cap, fill=(0,0,0), font=font)
        canvas.paste(crops[idx], (x0, y0+caption_h))

    canvas.save(out_path, "PNG")
    return out_path

jobs = [
    ("/mnt/data/2025-12-19_03-15-21.png",
     ["dallas_rustlers_1981-05-11.png","chicago_fire_1981-05-11.png","detroit_dukes_1981-05-11.png","minneapolis_blizzard_1981-05-11.png"],
     "/mnt/data/abl_division_collage_clean_3.png"),
    ("/mnt/data/2025-12-19_03-10-03.png",
     ["san_francisco_warriors_1981-05-11.png","los_angeles_cobras_1981-05-11.png","phoenix_firebirds_1981-05-11.png","san_diego_seraphs_1981-05-11.png"],
     "/mnt/data/abl_division_collage_clean_4.png"),
    ("/mnt/data/2025-12-19_03-12-42.png",
     ["boston_patriots_1981-05-11.png","new_york_aces_1981-05-11.png","philadelphia_fury_1981-05-11.png","pittsburgh_express_1981-05-11.png"],
     "/mnt/data/abl_division_collage_clean_5.png"),
    ("/mnt/data/2025-12-19_03-11-47.png",
     ["charlotte_colonels_1981-05-11.png","miami_hurricanes_1981-05-11.png","tampa_bay_storm_1981-05-11.png","atlanta_kings_1981-05-11.png"],
     "/mnt/data/abl_division_collage_clean_6.png")
]

outs=[]
for src,caps,out in jobs:
    outs.append(clean_collage(src, caps, out))
outs
