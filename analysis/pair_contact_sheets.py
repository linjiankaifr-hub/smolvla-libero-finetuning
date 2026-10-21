"""Make labeled contact sheets from audited real evaluation videos."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
out=Path(a.root)/'results/paired-evaluation-20261021'
summary=json.loads((out/'paired-summary.json').read_text())
assert summary['complete'] and summary['executions']==60
made=[]
for category in ['improved','both_failure','regressed','both_success']:
    if category not in summary['representatives']:continue
    row=summary['representatives'][category]
    image=Image.new('RGB',(800,650),'#f3f5f8');draw=ImageDraw.Draw(image)
    draw.text((12,8),f"Same official init {row['official_init_index']}, seed {row['seed']} | {category}",fill='#172333')
    for line,group in enumerate(['base','finetuned']):
        video=out/row[f'{group}_video']
        with imageio.get_reader(video) as reader:
            frames=[frame for frame in reader]
        steps=row[f'{group}_steps']
        assert len(frames)>=steps and len(frames)<=steps+2
        positions=[0,min(60,len(frames)-1),len(frames)-1]
        top=35+line*300
        flag=row[f'{group}_success']
        draw.text((12,top),f"{group} | final success={flag} | {steps} control steps",fill='#172333')
        for col,index in enumerate(positions):
            img=Image.fromarray(np.asarray(frames[index],dtype=np.uint8)).resize((256,256))
            image.paste(img,(8+col*264,top+18))
            label=f'Frame {index}' if col<2 else f'Final frame {index}'
            draw.text((8+col*264,top+276),label,fill='#172333')
    draw.text((12,636),'Real decoded simulation frames; final frames can represent different control times.',fill='#172333')
    filename=f'{category}-contact-sheet.png';image.save(out/filename);made.append(filename)
print(json.dumps({'contact_sheets':made}),flush=True)
