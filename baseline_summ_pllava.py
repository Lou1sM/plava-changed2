import numpy as np
import os
import json
from os.path import join
from tqdm import tqdm
import torch
from PIL import Image
from datasets import load_dataset
from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates
from torchvision.transforms import ToTensor

def get_all_testnames():
    with open('moviesumm_testset_names.txt') as f:
        official_names = f.read().split('\n')
    with open('clean-vid-names-to-command-line-names.json') as f:
        clean2cl = json.load(f)
    #assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
    assert all(x in official_names for x in clean2cl.keys())
    test_vidnames = list(clean2cl.values())
    return test_vidnames, clean2cl

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-beams', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=4)
parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
parser.add_argument('--min-len', type=int, default=600)
parser.add_argument('--max-len', type=int, default=650)
parser.add_argument('--vidname', type=str, default='the-sixth-sense_1999')
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--with-script', action='store_true')
parser.add_argument('--with-whisper-transcript', action='store_true')
parser.add_argument('--with-caps', action='store_true')
parser.add_argument('--mask-name', action='store_true')
parser.add_argument('--expdir-prefix', type=str, default='experiments')
parser.add_argument('--kf-dir-prefix', type=str, default='experiments')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
ARGS = parser.parse_args()

assert not (ARGS.with_whisper_transcript and ARGS.with_script)
llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
            'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            }
model, processor = load_pllava('llava-hf/llava-1.5-7b-hf', num_frames=ARGS.num_frames, use_lora=True, weight_dir=None, lora_alpha=32, pooling_shape=(16,8,8))
model.to(ARGS.device)
model.eval()

if ARGS.mask_name:
    assert ARGS.with_script or ARGS.with_whisper_transcript

ds = load_dataset("rohitsaxena/MovieSum")
test_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}
if ARGS.vidname != 'all':
    test_vidnames = [ARGS.vidname]

if ARGS.with_script:
    outdir = 'script-direct'
else:
    outdir = 'vidname-only'

outdir = os.path.join(ARGS.expdir_prefix, outdir)

erroreds = []
for vn in tqdm(test_vidnames):
    vid_subpath = f'moviesumm/{vn}'

    image_dir = join(ARGS.kf_dir_prefix, 'ffmpeg-keyframes', vid_subpath)
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    image_paths = image_paths[::len(image_paths) // ARGS.num_frames][:ARGS.num_frames]
    images = [ToTensor()(np.array(Image.open(fp))).half().to(model.device) for fp in image_paths]
    if os.path.exists(maybe_summ_path:=f'{outdir}/{vn}-summary.txt') and not ARGS.recompute:
        print(f'Summ already at {maybe_summ_path}')
        continue

    if ARGS.with_script:
        gt_match_name = cl2clean[vn]
        gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
        gt_script = gt_match['script']
        full_summarize_prompt = f'Based on the following script:\n{gt_script}\nsummarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'

    else:
        full_summarize_prompt = f'Summarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'
    conv = conv_templates['plain'].copy()
    n_beams = ARGS.n_beams
    min_len=600
    max_len=650
    summarize_prompt = full_summarize_prompt
    with torch.no_grad():
        summ_tokens = None
        for n_tries in range(8):
            try:
                conv._append_message(conv.roles[0], summarize_prompt)
                conv._append_message(conv.roles[1], None)
                response, _ = pllava_answer(conv=conv, model=model, processor=processor, do_sample=False, img_list=images, max_new_tokens=1, print_res=False)
                break
            except torch.cuda.OutOfMemoryError:
                summarize_prompt = summarize_prompt[:,5000:]
                max_len -= 50
                min_len -= 50
                print(f'OOM, reducing min,max to {min_len}, {max_len}')
        summ = response.strip()
        print(summ)
        os.makedirs(outdir, exist_ok=True)
        print('writing to', maybe_summ_path)
        with open(maybe_summ_path, 'w') as f:
            f.write(summ)

print(erroreds)
with open('errored.txt', 'w') as f:
    f.write('\n'.join(erroreds))
