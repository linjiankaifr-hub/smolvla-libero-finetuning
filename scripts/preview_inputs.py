import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from data_pipeline import ROOT

with h5py.File(ROOT / 'data/episodes.h5', 'r') as h5:
    source = h5['0']['observation.images.image'][0]
video = imageio.get_reader(ROOT / 'results/environment-smoke/episode_00.mp4')
rendered = video.get_data(0)
video.close()
canvas = Image.new('RGB', (768, 292), 'white')
draw = ImageDraw.Draw(canvas)
for i, (name, pixels) in enumerate([
    ('Dataset episode 0', source), ('Environment input: rotate180', rendered),
    ('Alternative: vertical flip', np.ascontiguousarray(rendered[:, ::-1]))]):
    canvas.paste(Image.fromarray(pixels).resize((256, 256)), (i*256, 36))
    draw.text((i*256+6, 8), name, fill='black')
canvas.save(ROOT / 'results/input-preview.jpg')
