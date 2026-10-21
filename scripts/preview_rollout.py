import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw

from data_pipeline import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('--output', default='results/rollout-preview.jpg')
args = parser.parse_args()
reader = imageio.get_reader(ROOT / args.video)
count = reader.count_frames()
indices = [0, count//3, count*2//3, count-1]
canvas = Image.new('RGB', (1024, 284), 'white')
draw = ImageDraw.Draw(canvas)
for i, index in enumerate(indices):
    frame = Image.fromarray(reader.get_data(index)).resize((256, 256))
    canvas.paste(frame, (i*256, 28))
    draw.text((i*256+6, 8), f'Control frame {index}', fill='black')
reader.close()
canvas.save(ROOT / args.output)
