"""
Drives run_iter.py over many collection episodes to scale the dataset past the
single-episode 468-frame baseline, cycling through weather presets and using
--randomize-spawn so each episode contributes a different (weather, route
position, off-center perturbation) combination.

Deliberately sticks to "standard" daytime/dusk weather presets for now -- night
and heavy-rain long-tail conditions are held back until the CBF controller is
stabilized (Priority 2), per the team's execution-order guidance.

Usage:
    python collect_dataset.py --host 192.168.56.1 --port 2000 --target-frames 50000

Sizing: an episode saves ~870 frames (every 5th sim tick over the ~4359-tick episode).
Wall-clock time per episode depends on server rendering speed, not sim time -- on this
project's dev server that was observed at ~5 sim-ticks/sec, i.e. ~14-15 min/episode, so
50k frames is roughly 58 episodes / ~14 hours, and 100k is roughly double. Budget for a
long-running background job, not a quick script.
"""
import argparse
import fnmatch
import glob
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RUN_ITER = os.path.join(SCRIPT_DIR, 'run_iter.py')

STANDARD_WEATHER = [
    'ClearNoon', 'CloudyNoon', 'WetNoon',
    'ClearSunset', 'CloudySunset', 'WetSunset',
]


def is_unc_path(path):
    return path.startswith('\\\\') or path.startswith('//')


def _configure_smb_if_needed(output_dir):
    if not is_unc_path(output_dir):
        return
    import smbclient
    username = os.environ.get('SMB_USERNAME')
    password = os.environ.get('SMB_PASSWORD')
    if not username or not password:
        raise RuntimeError('--output-dir is a UNC path but SMB_USERNAME/SMB_PASSWORD are not set.')
    smbclient.ClientConfig(username=username, password=password)


def _episode_dirs(output_dir, run_no):
    """Existing run<run_no>_ep<N>_images dir names (not full paths) under output_dir."""
    pattern = f'run{run_no}_ep*_images'
    if is_unc_path(output_dir):
        import smbclient
        _configure_smb_if_needed(output_dir)
        return [e for e in smbclient.listdir(output_dir) if fnmatch.fnmatch(e, pattern)]
    if not os.path.isdir(output_dir):
        return []
    return [os.path.basename(p) for p in glob.glob(os.path.join(output_dir, pattern))]


def frames_collected(output_dir, run_no):
    _configure_smb_if_needed(output_dir)
    total = 0
    for entry in _episode_dirs(output_dir, run_no):
        path = os.path.join(output_dir, entry)
        if is_unc_path(output_dir):
            import smbclient
            total += len(smbclient.listdir(path))
        else:
            total += len(glob.glob(os.path.join(path, '*.png')))
    return total


def next_episode_index(output_dir, run_no):
    """So re-running this script for another batch continues ep<N> numbering instead of
    restarting at ep0 and colliding with (silently overwriting frames in) an earlier batch."""
    indices = []
    prefix, suffix = f'run{run_no}_ep', '_images'
    for entry in _episode_dirs(output_dir, run_no):
        try:
            indices.append(int(entry[len(prefix):-len(suffix)]))
        except ValueError:
            continue
    return max(indices) + 1 if indices else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server IP')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--run-no', type=int, default=0,
                         help='run_iter.py --run_no to collect under (0 = expert-autopilot data '
                              'collection mode; leave at 0 unless you know you want NN+CBF-driven '
                              'episodes instead)')
    parser.add_argument('--target-frames', type=int, default=50000,
                         help='stop once this many frames have been collected across all episodes (default: 50000)')
    parser.add_argument('--base-seed', type=int, default=0,
                         help='episode i uses seed base-seed+i, for reproducibility (default: 0)')
    parser.add_argument('--max-episodes', type=int, default=500,
                         help='cap on episodes run in *this invocation* (default: 500) -- use this to run in '
                              'monitored batches: re-running the script later continues episode numbering from '
                              'where the last invocation left off (existing run<N>_ep*_images dirs are detected) '
                              'and checks the same --target-frames total, rather than restarting/overwriting')
    parser.add_argument('--weather-set', nargs='+', default=STANDARD_WEATHER,
                         help='weather presets to cycle through (default: standard daytime/dusk set)')
    parser.add_argument('--output-dir', default='.',
                         help='root directory for collected run<N>_ep*_images/video folders (default: current '
                              'directory). Point this at a mounted network share to avoid local disk usage.')
    parser.add_argument('--track', default='austin_map',
                         help="passed through to run_iter.py's --track (default: austin_map; pass "
                              "--track town04 for the original Phase 1 baseline map)")
    args = parser.parse_args()

    start_episode = next_episode_index(args.output_dir, args.run_no)
    if start_episode > 0:
        print(f'[collect_dataset] resuming: found existing episodes, continuing from ep{start_episode}')

    episode = start_episode
    episodes_run = 0
    while episodes_run < args.max_episodes:
        n_frames = frames_collected(args.output_dir, args.run_no)
        print(f'[collect_dataset] episode {episode}: {n_frames}/{args.target_frames} frames collected so far')
        if n_frames >= args.target_frames:
            print('[collect_dataset] target reached.')
            return

        weather = args.weather_set[episode % len(args.weather_set)]
        seed = args.base_seed + episode
        tag = f'ep{episode}'
        cmd = [
            sys.executable, RUN_ITER,
            '-r', str(args.run_no),
            '--host', args.host,
            '--port', str(args.port),
            '--weather', weather,
            '--seed', str(seed),
            '--episode-tag', tag,
            '--randomize-spawn',
            '--output-dir', args.output_dir,
            '--track', args.track,
        ]
        print(f'[collect_dataset] episode {episode}: weather={weather} seed={seed} tag={tag}')
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=False)
        episode += 1
        episodes_run += 1

    print(f'[collect_dataset] stopped after running {episodes_run} episode(s) this invocation '
          f'(--max-episodes={args.max_episodes}) -- '
          f'{frames_collected(args.output_dir, args.run_no)}/{args.target_frames} frames total')


if __name__ == '__main__':
    main()
