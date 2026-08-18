"""Converts a TUM racetrack-database centerline+width CSV (x_m,y_m,w_tr_right_m,w_tr_left_m)
into a minimal closed-loop OpenDRIVE (.xodr) road, loadable via
carla.Client.generate_opendrive_world(). See ../racetrack_source/ for the source data and its
LGPLv3 license (https://github.com/TUMFTM/racetrack-database).

The whole corridor (w_tr_left + w_tr_right) becomes a single driving lane -- there's only one
ego vehicle and no oncoming traffic, so there's no need to split it into directional lanes.

Usage:
    python tools/track_to_opendrive.py ../racetrack_source/Austin.csv ../tracks/austin.xodr --name Austin
"""
import argparse
import os

import numpy as np


def build_opendrive(points, widths, name):
    """points: (N,2) closed-loop centerline [x_m, y_m]. widths: (N,) total corridor width per point.
    Returns (xodr_string, total_length). One <line> geometry per point-to-point segment, each with
    its own explicit x/y/hdg -- no heading-continuity math needed -- and a single driving lane whose
    width steps at each point via <width sOffset a=width b=c=d=0> breakpoints."""
    n = points.shape[0]
    deltas = np.roll(points, -1, axis=0) - points  # segment i: points[i] -> points[(i+1) % n], closing the loop
    seg_lengths = np.linalg.norm(deltas, axis=1)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    s_stations = np.concatenate([[0.0], np.cumsum(seg_lengths)[:-1]])
    total_length = float(np.sum(seg_lengths))

    geometries = []
    for i in range(n):
        x, y = points[i]
        geometries.append(
            f'      <geometry s="{s_stations[i]:.6f}" x="{x:.6f}" y="{y:.6f}" '
            f'hdg="{headings[i]:.8f}" length="{seg_lengths[i]:.6f}"><line/></geometry>'
        )

    width_entries = []
    for i in range(n):
        w = max(float(widths[i]), 1.0)  # floor width so no degenerate/zero-width lane
        width_entries.append(
            f'          <width sOffset="{s_stations[i]:.6f}" a="{w:.4f}" b="0" c="0" d="0"/>'
        )

    xodr = f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="{name}" version="1.00" north="0" south="0" east="0" west="0"/>
  <road name="{name}" length="{total_length:.6f}" id="1" junction="-1">
    <link>
      <predecessor elementType="road" elementId="1" contactPoint="end"/>
      <successor elementType="road" elementId="1" contactPoint="start"/>
    </link>
    <planView>
{chr(10).join(geometries)}
    </planView>
    <elevationProfile>
      <elevation s="0" a="0" b="0" c="0" d="0"/>
    </elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0">
        <center>
          <lane id="0" type="none" level="false">
            <roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.13"/>
          </lane>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
{chr(10).join(width_entries)}
            <roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.13"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
'''
    return xodr, total_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', help='TUM racetrack-database tracks/<Name>.csv')
    parser.add_argument('output_xodr', help='output .xodr path')
    parser.add_argument('--name', default=None, help='road name (default: input filename stem)')
    args = parser.parse_args()

    data = np.loadtxt(args.input_csv, delimiter=',', skiprows=1)
    points = data[:, :2]
    widths = data[:, 2] + data[:, 3]
    name = args.name or os.path.splitext(os.path.basename(args.input_csv))[0]

    xodr, total_length = build_opendrive(points, widths, name)
    out_dir = os.path.dirname(args.output_xodr)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_xodr, 'w') as f:
        f.write(xodr)
    print(f'Wrote {args.output_xodr}: {points.shape[0]} segments, {total_length:.1f}m total length')


if __name__ == '__main__':
    main()
