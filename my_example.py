import os
from os.path import join
from tqdm import tqdm
import pandas as pd
import logging
import argparse
import json
from natsort import natsorted
import numpy as np
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor
from decord import VideoReader, cpu
from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates

# Configure logging
logging.getLogger().setLevel(logging.ERROR)

# Load dataset
with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

show_name_dict = {
    'friends':'Friends',
    'house': 'House M.D.',
    'met': 'How I Met You Mother',
    'bbt': 'The Big Bang Theory',
    'castle': 'Castle',
    'grey': "Grey's Anatomy",
}

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, resolution=336):
    transforms = torchvision.transforms.Resize(size=resolution)
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    return [Image.fromarray(vr[i].asnumpy()) for i in frame_indices]

def get_texts(split_name, vid_subpath):
    scenes = []
    for fn in natsorted(os.listdir(stexts_rag_caches_dir:=join(ARGS.rag_caches_prefix, 'rag-caches', split_name, vid_subpath, 'scene_texts'))):
        with open(join(stexts_rag_caches_dir, fn)) as f:
            scenes.append(f.read())
    return scenes

def get_showseaseps(show_name_, seas_num_, ep_num_):
    showseaseps = []
    if show_name_=='all':
        show_names_to_compute = natsorted(os.listdir(join(ARGS.rag_caches_prefix, 'rag-caches', 'ours', 'tvqa/')))
        show_names_to_compute = [x for x in show_names_to_compute if x!='bbt']
    else:
        show_names_to_compute = [show_name_]
    for show_name in show_names_to_compute:
        if seas_num_ == -1:
            seass_to_compute = natsorted([int(fn[7:]) for fn in os.listdir(join(ARGS.rag_caches_prefix, f'rag-caches', 'ours', 'tvqa', show_name))])
        else:
            seass_to_compute = [seas_num_]

        for seas_num in seass_to_compute:
            if ep_num_ == -1:
                for fn in natsorted(os.listdir(join(ARGS.rag_caches_prefix, f'rag-caches', 'ours', 'tvqa', show_name, f'season_{seas_num}'))):
                    ep_num = int(fn[8:].removesuffix('.mp4'))
                    showseaseps.append((show_name, seas_num, ep_num))
            else:
                showseaseps.append((show_name, seas_num, ep_num_))
    return showseaseps

def answer_qs(show_name, season, episode, model, processor, ep_qs):
    dset_name = 'tvqa'
    vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'
    scenes = get_texts('ours', vid_subpath)
    scene_text = '[SCENE_BREAK]'.join('\n'.join(l for l in s) for s in scenes)
    scene_text = scene_text[-ARGS.prompt_prefix:]

    image_dir = f'../amazon_video/data/ffmpeg-keyframes/{vid_subpath}'
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]
    pick_every = len(image_paths) // ARGS.num_frames
    image_paths = image_paths[::pick_every][:ARGS.num_frames]
    assert len(image_paths) == ARGS.num_frames

    transform = ToTensor()
    images = [Image.open(fp) for fp in image_paths]
    images = [transform(np.array(x)).half() for x in images]
    images = [x.to(model.device) for x in images]

    n_correct = 0
    conv = conv_templates['plain'].copy()

    for i, qdict in enumerate(ep_qs['questions']):
        qsent = qdict['q']
        options = '\n'.join(f"{idx}: {qdict[f'a{idx}']}" for idx in range(5))

        prompt = f"""Context from the show:{scene_text}Question: {qsent}Options:{options}Answer with ONLY the number (0-4) of the correct option, nothing else."""

        conv._append_message(conv.roles[0], prompt)
        conv._append_message(conv.roles[1], None)

        #try:
        response, _ = pllava_answer(
                conv=conv,
                model=model,
                processor=processor,
                do_sample=False,
                img_list=images,
                max_new_tokens=1,
                print_res=False
        )

        # Extract answer number
        output = response.strip()
        try:
            ans = int(output[0])  # Get first character
            if 0 <= ans <= 4:
                if ans == qdict['answer_idx']:
                    n_correct += 1
                if ARGS.verbose:
                    print(f"Question: {qsent}")
                    print(f"Options: {options}")
                    print(f"Model output: {output}")
                    print(f"Predicted: {ans}, Correct: {qdict['answer_idx']}\n")
        except (ValueError, IndexError):
            if ARGS.verbose:
                print(f"Failed to parse answer: {output}")

        print(ans, n_correct)
        #except Exception as e:
        #    print(f"Error generating answer: {e}")
        #    continue

    n = len(ep_qs["questions"])
    print(f'VQA accuracy: {n_correct}/{n} = {n_correct/n:.5f}')
    return n_correct, n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, default=-1)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--num-frames', type=int, default=4, help='Number of frames to sample from video')
    parser.add_argument('--resolution', type=int, default=336, help='Frame resolution')
    parser.add_argument('--prompt-prefix', type=int, default=336, help='Frame resolution')
    parser.add_argument("--rag-caches-prefix", type=str, default='.')
    ARGS = parser.parse_args()

    # Initialize PLLaVA model
    model, processor = load_pllava(
        'llava-hf/llava-1.5-7b-hf',
        num_frames=ARGS.num_frames,
        use_lora=True,
        weight_dir=None,
        lora_alpha=32,
        pooling_shape=None
    )

    tot_n_correct, tot = 0, 0
    all_scores = []
    os.makedirs(out_dir:=f'tvqa-results/pllava', exist_ok=True)
    showseaseps = get_showseaseps(ARGS.show_name, ARGS.season, ARGS.ep)

    for show_name, seas, ep in (pbar:=tqdm(showseaseps)):
        season_qs = full_dset_qs[show_name_dict[show_name]][f'season_{seas}']
        if f'episode_{ep}' not in season_qs.keys():
            continue
        if (show_name, seas, ep) == ('house', 4, 11):
            continue

        ep_qs = season_qs[f'episode_{ep}']
        cache_fp = os.path.join(out_dir, f'{show_name}_s{seas:01}e{ep:01}.json')

        if os.path.exists(cache_fp) and not ARGS.recompute:
            with open(cache_fp) as f:
                x = f.read().split()
            new_correct, new_tot = int(x[0]), int(x[1])
        else:
            new_correct, new_tot = answer_qs(show_name, seas, ep, model, processor, ep_qs)
            with open(cache_fp, 'w') as f:
                f.write(f'{new_correct} {new_tot}')

        tot_n_correct += new_correct
        tot += new_tot
        all_scores.append([show_name, seas, ep, new_correct, new_tot, new_correct/new_tot])
        pbar.set_description(f'{show_name}-s{seas}e{ep}, running avg: {tot_n_correct}/{tot}={tot_n_correct/tot}')

    df = pd.DataFrame(all_scores, columns=['show', 'season', 'episode', 'n_correct', 'n', 'acc'])
    print(df.drop('show', axis=1).mean(axis=0))
    df.to_csv(f'{out_dir}/{ARGS.show_name}_{ARGS.season}-tvqa-results.csv')
