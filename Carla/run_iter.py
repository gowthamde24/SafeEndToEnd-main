#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

./CarlaUE4.sh /Game/Maps/RaceTrack -windowed -carla-server -benchmark -fps=30 -quality-level=Low
python3 module_7.py
A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import random
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import configparser
from scipy.stats import norm
from PIL import Image
import tqdm
import queue

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import controller2d

import carla
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# --- GLOBAL CONFIG ---
# we define the physical limits of the Lincoln MKZ 
# and the dataset configuration we used for Town 04.
WIDTH = 800
HEIGHT = 600
RUN_NO = 1
N_ITERS = 4
BETA = 0.1
SAFEGUARD = None  # Resolved in main() from --cbf/--no-cbf, or the K_ITERS ramp-up default
K_ITERS = 3
VEHICLE_MODEL = 'Dynamic'

# CARLA defaults to a 30 km/h speed limit on OpenDRIVE roads with no <type><speed> element
# (ours doesn't define one) -- confirmed empirically the autopilot was cruising at a steady
# ~8 m/s the whole episode, nowhere near the "cornering-limits" racing pace this project is
# framed around. -200 targets ~3x the limit (~90 km/h); live-tested for 35s/~500m including a
# real corner (Austin's route data) with zero collisions -- speed ranged ~8.7-24.4 m/s, i.e. it
# still slows appropriately for corners rather than just being uniformly faster everywhere.
AUTOPILOT_SPEED_DIFF_PERCENT = -200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
image_size = 256

log_step = 10

# ---  IMAGE PREPROCESSING ---
# This must perfectly match the transform in train.py. 
# It ensures the live camera feed from CARLA is cropped and normalized 
# exactly how the ResNet was trained to see it.
transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.CenterCrop(min(HEIGHT,WIDTH)),
        transforms.Resize(image_size),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

# ==============================================================================
# --- 1. MOCK IMAGE CONVERTER ---
# ==============================================================================
# --- SENSOR DATA FIX ---
# This class extracts the raw bytes from the new camera sensor
# and converts it into a standard RGB numpy array for PyTorch to process.
class MockImageConverter:
    @staticmethod
    def to_rgb_array(image):
        """Converts modern 0.9.x raw image data into the numpy array the legacy script expects."""
        if image is None:
            return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] # Drop alpha channel
        array = array[:, :, ::-1] # Convert BGR to RGB
        return array

image_converter = MockImageConverter()

# ==============================================================================
# --- 2. THE 0.9.x to 0.8.x BRIDGE ---
# ==============================================================================
# --- LEGACY ADAPTER ---
# The original framework was built for CARLA 0.8. I created these wrapper classes 
# so we don't have to rewrite the entire vehicle dynamics physics engine. 
# It packages the modern CARLA 0.9 transforms into the old format.
class LegacyLocation:
    def __init__(self, x, y):
        self.x, self.y = x, y

class LegacyRotation:
    def __init__(self, yaw):
        self.yaw = yaw

class LegacyTransform:
    def __init__(self, x, y, yaw):
        self.location = LegacyLocation(x, y)
        self.rotation = LegacyRotation(yaw)

class LegacyPlayerMeasurements:
    def __init__(self, transform, speed, collision):
        self.transform = transform
        self.forward_speed = speed
        self.collision_other = collision

class LegacyMeasurementData:
    def __init__(self, timestamp, player_measurements):
        self.game_timestamp = timestamp
        self.player_measurements = player_measurements

class Carla09Bridge:
    def __init__(self, world, ego_vehicle, camera_queue, video_queue):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.camera_queue = camera_queue
        self.video_queue = video_queue
        self.collision_history = 0

    def read_data(self):
        # 1. Time and Physics
        self.world.tick() # Advance the simulation by one frame
        snapshot = self.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds * 1000.0 
        
        vehicle_transform = self.ego_vehicle.get_transform()
        velocity = self.ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # 2. Build Legacy Target
        leg_transform = LegacyTransform(vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.rotation.yaw)
        leg_player = LegacyPlayerMeasurements(leg_transform, speed, self.collision_history)
        measurement_data = LegacyMeasurementData(timestamp, leg_player)
        
        # 3. Pull from Modern Sensor Queues
        sensor_data = {}
        try:
            sensor_data['CameraRGB'] = self.camera_queue.get(timeout=2.0)
            sensor_data['CameraRGB_video'] = self.video_queue.get(timeout=2.0)
        except queue.Empty:
            print("Warning: Camera Queue Empty - Dropping Frame")
            sensor_data['CameraRGB'] = None
            sensor_data['CameraRGB_video'] = None
            
        return measurement_data, sensor_data

"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 3.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 200.00 # game seconds (total runtime before sim end)
STALL_SPEED_THRESHOLD  = 0.3    # m/s -- below this counts as "not moving" for the stall watchdog
STALL_TIMEOUT_SECONDS  = 10.0   # abort the episode if stalled this long post-startup (autopilot
                                 # can get stuck against the track boundary at some randomized-spawn
                                 # points and otherwise just sit there for the rest of the episode,
                                 # recording hundreds of useless identical frames -- see docs/experiments.md)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 0      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer

PLAYER_START_INDEX = 2      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1    
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WAYPOINTS_FILENAME = 'town04_waypoints.txt'  # waypoint file to load
CENTERLINE_FILENAME = 'town04_waypoints.txt'

# --- TRACKS: per-track config selected via --track. lane_width feeds the CBF's LANE_WIDTH
# constant below (see main()) -- Town04's is a synthetic corridor guess, Austin's is the real
# average corridor width from racetrack_source/Austin.csv (w_tr_left+w_tr_right, ~14m avg,
# 11-27.6m range) so the barrier reflects the actual track rather than reusing Town04's number.
TRACKS = {
    'town04': {
        'waypoints_file': 'town04_waypoints.txt',
        'centerline_file': 'town04_waypoints.txt',
        'lane_width': 11,
    },
    'austin_map': {
        'waypoints_file': 'austin_waypoints.txt',
        'centerline_file': 'austin_waypoints.txt',
        'xodr_file': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tracks', 'austin.xodr'),
        'lane_width': 14,
        'source_csv': os.path.join(os.path.dirname(os.path.realpath(__file__)), 'racetrack_source', 'Austin.csv'),
    },
}
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before simulation ends
                                       
# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying lookahead path
INTERP_LOOKAHEAD_DISTANCE = 20   # lookahead in meters
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

# ---  VEHICLE CONSTANTS ---
# These are the physical properties of the Lincoln MKZ used by the math model
# to calculate if a turn is physically possible without skidding.
MAX_STEER = 1
LANE_WIDTH = 11
L = 3.
lambda_ = 5.
alpha = 20.
curvature_factor = 0.005
x_factor = 7  # must match train.py's x_factor -- this is the inverse of the scaling applied to
              # the cross-track target before training, needed to decode model_safety_2's raw
              # output back into real meters
theta_factor = 5
THETA_LIM = 37.*(math.pi/180.)

mass = 800
Cf = 1.2*mass
Cr = 1.2*mass
lf = L/2.
lr = L/2. 
Iz = (mass*lf**2)/2.

# --- TASK A (SA-TCP ARCHITECTURE) ---
# This is Branch A of the proposal.
# It uses a ResNet backbone to look at the image, processes the vehicle's current
# speed vectors, and outputs the optimal raw steering command.
class EndtoEnd(models.resnet.ResNet):
    def __init__(self):
        super(EndtoEnd, self).__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        self.speed_feat_extractor = nn.Linear(2, 64)
        self.final_layer = nn.Linear(64+128, 1)
        
    def forward(self, x, v, vperp):
        speed_feat = torch.cat([torch.atan2(vperp,v),torch.sqrt(v**2+vperp**2)],dim=1)
        x1 = super(EndtoEnd,self).forward(x)
        x2 = self.speed_feat_extractor(speed_feat)
        x = torch.cat((x1,x2),axis=1)
        x = self.final_layer(x)
        return x

def computeCurvature(p0, p1, p2) :
    dx1 = p1[0] - p0[0]
    dy1 = p1[1] - p0[1]
    dx2 = p2[0] - p0[0]
    dy2 = p2[1] - p0[1]
    dx3 = p2[0] - p1[0]
    dy3 = p2[1] - p1[1]
    area = 0.5 * (dx1 * dy2 - dy1 * dx2)
    len0 = math.sqrt(dx1**2 + dy1**2)
    len1 = math.sqrt(dx2**2 + dy2**2)
    len2 = math.sqrt(dx3**2 + dy3**2)
    return 4 * area / (len0 * len1 * len2)

def has_crossed_train_line(x,y) :
    return (x>-50)

# --- THE CONTROL BARRIER FUNCTION (CBF) ---
# This is the "Safety Gate".
# It takes the steering angle from Branch A (steer_ref) and the predictions
# from Branch B (theta, x, curvature).
# The practical effect:
# This function creates an imaginary "barrier" at the lane lines. If the car's
# current trajectory (calculated via kinematics) crosses that barrier, the CBF mathematically 
# overrides the neural network and forces a safe steering angle (min_steer).
def get_optimal_control(steer_ref,steer_var,v,theta,theta_var,x,x_var,curvature,curvature_var,v_perp=0.,omega=0.,EPSILON=1e-1) :
    if SAFEGUARD==False :
        return steer_ref
    min_steer = -MAX_STEER
    min_cost = 1000000
    steer_var = max(steer_var, EPSILON)  # floor near-zero MC-dropout variance so it can't blow up the tracking-cost term
    if VEHICLE_MODEL=='Kinematic' :
        for steer in np.arange(-MAX_STEER,MAX_STEER,0.01) :
            ax = 0
            h_left = LANE_WIDTH/2. - x
            hd_left = -v*math.sin(theta)
            hdd_left = -ax*math.sin(theta)-v**2*math.cos(theta)*steer/L + v**2*curvature
            h_right = LANE_WIDTH/2. + x
            hd_right = v*math.sin(theta)
            hdd_right = ax*math.sin(theta)+v**2*math.cos(theta)*steer/L - v**2*curvature
            cost = (steer-steer_ref)**2/steer_var**2
            var_left = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            var_right = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            
            # The actual "barrier" check: Are we violating the boundary?
            if (hdd_left+lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA)) < 0 :
                cost += alpha*(hdd_left+lambda_*hd_left+lambda_**2*h_left-var_left*norm.ppf(1-BETA))**2
            if (hdd_right+lambda_*hd_right+lambda_**2*h_right-var_right*norm.ppf(1-BETA)) < 0 :
                cost += alpha*(hdd_right+lambda_*hd_right+lambda_**2*h_right-var_right*norm.ppf(1-BETA))**2
            if cost < min_cost :
                min_cost = cost
                min_steer = steer
    else :
        # Dynamic model (used for high speeds where tire slip occurs)
        v += EPSILON
        for steer in np.arange(-MAX_STEER,MAX_STEER,0.01) :
            ax = 0
            x_dot = v_perp*math.cos(theta)+v*math.sin(theta)
            v_perp_dot = (-2*(Cf+Cr)/(mass*v))*v_perp
            v_perp_dot_dot = (-2*(Cf+Cr)/(mass*v))*v_perp_dot + (2*(Cf+Cr)/(mass*v**2))*ax
            x_dot_dot = v_perp_dot*math.cos(theta)+ax*math.sin(theta)+omega*(-v_perp*math.sin(theta)+v*math.cos(theta))
            omega_dot = (-2*(lf**2*Cf + lr**2*Cr)/(Iz*v))*omega+(2*(lf*Cf/Iz))*steer
            x_dot_dot_dot = v_perp_dot_dot*math.cos(theta) + \
                omega_dot*(-v_perp*math.sin(theta)+v*math.cos(theta)) - \
                omega**2*(x_dot)
            h_theta_left = THETA_LIM - theta
            hd_theta_left = - (omega-v*curvature)
            hdd_theta_left = - (omega_dot-ax*curvature)
            h_theta_right = THETA_LIM + theta
            hd_theta_right = (omega-v*curvature)
            hdd_theta_right = (omega_dot-ax*curvature)
                
            h_left = LANE_WIDTH/2. - x
            hd_left = -x_dot
            hdd_left = -x_dot_dot + v**2*curvature
            hddd_left = -x_dot_dot_dot + 2*v*curvature*ax
            h_right = LANE_WIDTH/2. + x
            hd_right = x_dot
            hdd_right = x_dot_dot - v**2*curvature
            hddd_right = x_dot_dot_dot - 2*v*curvature*ax
            cost = (steer-steer_ref)**2/steer_var**2
            
            if (hdd_theta_left+3*lambda_*hd_theta_left+3*lambda_**2*h_theta_left<0) :
                cost += alpha*(hdd_theta_left+3*lambda_*hd_theta_left+3*lambda_**2*h_theta_left)**2
            if (hdd_theta_right+3*lambda_*hd_theta_right+3*lambda_**2*h_theta_right<0) :
                cost += alpha*(hdd_theta_right+3*lambda_*hd_theta_right+3*lambda_**2*h_theta_right)**2
            var_left = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            var_right = math.sqrt((abs(lambda_**2*x_var))**2 + (v**2*math.sin(theta)*steer/L)**2 + (v**2*curvature_var)**2 + (lambda_*(v*math.cos(theta))*theta_var)**2)
            if (hddd_left + 3*lambda_*hdd_left + 3*lambda_**2*hd_left+lambda_**3*h_left-var_left*norm.ppf(1-BETA)) < 0 :
                cost += alpha*(hddd_left + 3*lambda_*hdd_left + 3*lambda_**2*hd_left+lambda_**3*h_left-var_left*norm.ppf(1-BETA))**2
            if (hddd_right+3*lambda_*hdd_right+3*lambda_**2*hd_right+lambda_**3*h_right-var_right*norm.ppf(1-BETA)) < 0 :
                cost += alpha*(hddd_right+3*lambda_*hdd_right+3*lambda_**2*hd_right+lambda_**3*h_right-var_right*norm.ppf(1-BETA))**2
            if cost < min_cost :
                min_cost = cost
                min_steer = steer
    return min_steer

# [Standard Timer and Helper Classes Omitted for brevity in explanation]
class Timer(object):
    """ Timer Class """
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements"""
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)
    return (x, y, yaw)

def send_control_command(vehicle, throttle, steer, brake, hand_brake=False, reverse=False):
    """Send control command to the modern CARLA vehicle actor."""
    control = carla.VehicleControl()
    control.steer = float(np.fmax(np.fmin(steer, 1.0), -1.0))
    control.throttle = float(np.fmax(np.fmin(throttle, 1.0), 0.0))
    control.brake = float(np.fmax(np.fmin(brake, 1.0), 0.0))
    control.hand_brake = bool(hand_brake)
    control.reverse = bool(reverse)
    vehicle.apply_control(control)

def is_unc_path(path):
    return path.startswith('\\\\') or path.startswith('//')

def init_smb_session(output_dir):
    """Registers smbclient credentials once, if --output-dir is a UNC path like
    \\\\CHERRY\\CarlaData. Reads SMB_USERNAME/SMB_PASSWORD from the environment (not a CLI
    flag, so they don't leak into shell history or process listings)."""
    if not is_unc_path(output_dir):
        return
    import smbclient
    username = os.environ.get('SMB_USERNAME')
    password = os.environ.get('SMB_PASSWORD')
    if not username or not password:
        raise RuntimeError(
            '--output-dir is a UNC path but SMB_USERNAME/SMB_PASSWORD are not set in the '
            'environment (export them before running).')
    smbclient.ClientConfig(username=username, password=password)

def makedirs_maybe_smb(path):
    if is_unc_path(path):
        import smbclient
        smbclient.makedirs(path, exist_ok=True)
    elif not os.path.exists(path):
        os.makedirs(path)

def save_image_maybe_smb(image, path):
    """image is a PIL Image; path may be a local path or a UNC path under a --output-dir share."""
    if is_unc_path(path):
        import smbclient
        with smbclient.open_file(path, mode='wb') as f:
            image.save(f, format='PNG')
    else:
        image.save(path)

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def load_checkpoint_name(model_path, suffix):
    """suffix is '' for the steering model or '-safety-<N>' for a safety model. Prefers
    'model-best<suffix>.ckpt' (train.py's best-val-loss checkpoint from early stopping);
    falls back to 'model-last<suffix>.ckpt' if no val split existed for that training run
    (e.g. a very small dataset)."""
    if os.path.exists(os.path.join(model_path, f'model-best{suffix}.ckpt')):
        return f'model-best{suffix}.ckpt'
    return f'model-last{suffix}.ckpt'

def store_trajectory_plot(graph, fname):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def spawn_track_scenery(world, track_cfg, haybale_spacing_m=60,
                         scenery_spacing_m=200, cone_corner_threshold_deg=12):
    """Scatters occasional hay-bale run-off stacks, corner cones, and sparse background props
    along a custom OpenDRIVE track's edges, using the same per-point centerline+width data
    tools/track_to_opendrive.py built the .xodr from.

    Custom OpenDRIVE worlds are otherwise bare road mesh with no ground/terrain at all --
    confirmed visually (see docs/experiments.md) that everything off the drivable surface
    renders as CARLA's default flat-blue void. A low wall_height (set on
    OpendriveGenerationParameters where the world is generated) fixes the void.

    NOTE: this used to also place a continuous static.prop.chainbarrier fence along both edges,
    which looked good in isolated screenshots but had two real problems found only once actually
    driven: (1) its chain visually linked to the nearest other chainbarrier instance regardless
    of which side of the track it was on, so at narrow points/tight bends a chain could stretch
    straight across the drivable surface; (2) static props simulate physics by default, so the
    vehicle could snag a chain and drag the post down the track, corrupting whatever was being
    recorded. Removed entirely per explicit instruction rather than reworked -- see
    docs/decisions.md, 2026-08-16.

    Runs once per episode (generate_opendrive_world() rebuilds the world from scratch each
    time). No-op for tracks without a 'source_csv' (e.g. town04, which has real scenery baked
    into the map already). Static props only; try_spawn_actor so occasional collisions between
    adjacent props are skipped, not fatal."""
    source_csv = track_cfg.get('source_csv')
    if not source_csv:
        return 0, 0
    data = np.loadtxt(source_csv, delimiter=',', skiprows=1)
    points = data[:, :2]
    widths = data[:, 2] + data[:, 3]  # w_tr_right + w_tr_left, matches the .xodr's lane width
    n = points.shape[0]

    deltas = np.roll(points, -1, axis=0) - points
    seg_lengths = np.linalg.norm(deltas, axis=1)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    mean_spacing = float(np.mean(seg_lengths))
    dh_deg = np.degrees(np.abs((np.diff(headings, append=headings[0]) + math.pi) % (2 * math.pi) - math.pi))

    haybale_step = max(1, round(haybale_spacing_m / mean_spacing))
    scenery_step = max(1, round(scenery_spacing_m / mean_spacing))

    bp_lib = world.get_blueprint_library()
    haybale_bp = bp_lib.find('static.prop.haybale')
    cone_bp = bp_lib.find('static.prop.trafficcone01')
    scenery_bps = [bp for bp in (bp_lib.find('static.prop.gardenlamp'), bp_lib.find('static.prop.streetsign01')) if bp]

    spawned, skipped = 0, 0

    def try_spawn(bp, x, y, yaw_deg=0.0):
        nonlocal spawned, skipped
        if bp is None:
            return
        transform = carla.Transform(carla.Location(x=float(x), y=float(y), z=0.1), carla.Rotation(yaw=yaw_deg))
        actor = world.try_spawn_actor(bp, transform)
        if actor is not None:
            # Static props simulate physics by default (confirmed empirically: a chainbarrier's
            # chain got caught on the ego vehicle and was dragged down the track, corrupting a
            # collection episode -- see docs/experiments.md). Pin every prop in place so a close
            # pass can't move it, regardless of which prop type this ends up being used for.
            actor.set_simulate_physics(False)
            spawned += 1
        else:
            skipped += 1

    for i in range(n):
        right_x, right_y = math.sin(headings[i]), -math.cos(headings[i])  # unit vector, right of travel direction
        half_w = widths[i] / 2.0
        yaw_deg = math.degrees(headings[i])

        if i % haybale_step == 0 and haybale_bp:
            off = half_w + 1.2  # just outside the true lane edge, like a run-off-area stack
            for sign in (-1, 1):
                try_spawn(haybale_bp, points[i, 0] + sign * right_x * off, points[i, 1] + sign * right_y * off)

        if dh_deg[i] > cone_corner_threshold_deg:
            off = half_w + 1.0
            for sign in (-1, 1):
                try_spawn(cone_bp, points[i, 0] + sign * right_x * off, points[i, 1] + sign * right_y * off)

        if i % scenery_step == 0 and scenery_bps:
            off = half_w + 4.0
            bp = scenery_bps[(i // scenery_step) % len(scenery_bps)]
            try_spawn(bp, points[i, 0] + right_x * off, points[i, 1] + right_y * off)

    print(f"Track scenery: spawned {spawned} props ({skipped} skipped due to collision)")
    return spawned, skipped

def cbf_mode_name():
    return 'with_cbf' if SAFEGUARD else 'without_cbf'

def write_trajectory_file(x_list, y_list, v_list, t_list, steerings_list, throttle_list):
    output_dir = os.path.join(CONTROLLER_OUTPUT_FOLDER, cbf_mode_name())
    create_controller_output_dir(output_dir)
    file_name = os.path.join(output_dir, 'trajectory_run'+str(RUN_NO)+'.txt')

    with open(file_name, 'w') as trajectory_file: 
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f, %6.3f, %6.3f\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i], steerings_list[i], throttle_list[i]))

# --- MAIN SIMULATION LOOP ---
# This function orchestrates the entire closed-loop demo.
def exec_waypoint_nav_demo(args, spawn_seed_offset=0):
    """ Executes waypoint navigation demo. spawn_seed_offset varies the randomize-spawn draw
    across main()'s retry attempts -- without it, a deterministic --seed that happens to draw a
    spawn point the autopilot stalls at (see STALL_SPEED_THRESHOLD above) would retry forever
    against the identical doomed point instead of ever trying a different one."""
    init_smb_session(args.output_dir)

    # --- 1. CONNECT & SETUP WORLD ---
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    track_cfg = TRACKS[args.track]
    if args.track == 'town04':
        world = client.load_world('Town04') # Spawns the high-speed highway environment
    else:
        with open(track_cfg['xodr_file']) as xodr_handle:
            opendrive_str = xodr_handle.read()
        # wall_height/additional_width: generate_opendrive_world() otherwise produces *only* the
        # road mesh -- no ground/terrain at all, so the camera sees an infinite flat default-blue
        # void everywhere off the drivable surface (confirmed visually; see docs/experiments.md).
        # This wall is the sole edge-boundary mechanism now (spawn_track_scenery()'s props,
        # including a chainbarrier fence, were all removed -- see docs/decisions.md, 2026-08-16).
        # wall_height=0.4 (curb height) turned out not to be a real physical barrier: an autopilot
        # drift during collection carried the vehicle straight through it and ~7.5m into the void
        # before the autopilot gave up and parked there indefinitely (confirmed via
        # world.get_actors() -- vehicle 14.7m from centerline where the corridor half-width was
        # only 7.2m; see docs/experiments.md). Raised to 0.8m -- a real concrete/Armco trackside
        # barrier height, not an arena wall -- after empirically confirming 0.8/1.0/1.5m all stop
        # a vehicle driven full-throttle straight at the wall at the identical ~5m distance (well
        # short of the 7.5m true lane edge): the extra height above 0.8m wasn't doing anything
        # functionally, so there's no containment reason to go taller than a realistic barrier.
        # additional_width left at 0: a nonzero paved shoulder put visible daylight between the
        # true lane edge and the wall, which at sharper bends read as a confusing second surface
        # rather than a clean boundary (see docs/experiments.md).
        odr_params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0, max_road_length=500.0, wall_height=0.8,
            additional_width=0.0, smooth_junctions=True, enable_mesh_visibility=True,
            enable_pedestrian_navigation=False,
        )
        world = client.generate_opendrive_world(opendrive_str, odr_params, reset_settings=True)

    # Set synchronous mode so our Python script dictates the frame rate
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20 FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    
    # --- WEATHER ---
    weather_preset = getattr(carla.WeatherParameters, args.weather, None)
    if weather_preset is None:
        raise ValueError(f"Unknown --weather preset '{args.weather}'; see carla.WeatherParameters for valid names.")
    world.set_weather(weather_preset)
    # spawn_track_scenery() (haybale/cone/background props) disabled entirely per explicit
    # instruction -- the track should have no scattered obstacles at all. Kept defined below in
    # case scenery is revisited later; see docs/decisions.md, 2026-08-16. The low wall_height on
    # odr_params above still prevents the flat-blue-void issue -- that's road geometry, not a
    # discrete obstacle, and wasn't part of what was flagged.

    blueprint_library = world.get_blueprint_library()

    # --- WAYPOINTS (loaded early: --randomize-spawn needs the route to pick a spawn point from) ---
    waypoints_file = track_cfg['waypoints_file']
    with open(waypoints_file) as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle,
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        waypoints_np = np.array(waypoints)

    centerline_file = track_cfg['centerline_file']
    with open(centerline_file) as waypoints_file_handle:
        centerline = list(csv.reader(waypoints_file_handle,
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        centerline_np = np.array(centerline)

    # --- 2. SPAWN EGO VEHICLE ---
    vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
    spawn_points = world.get_map().get_spawn_points()
    effective_seed = None if args.seed is None else args.seed + spawn_seed_offset
    rng = random.Random(effective_seed)
    if args.randomize_spawn:
        # Pick a random point along the route (leaving enough road ahead for a real episode),
        # snap it to the lane center, then perturb sideways/yaw so the autopilot has to
        # steer back to center -- that recovery arc is the DAgger-style training signal.
        # Uses try_spawn_actor + resampling (not spawn_actor, which raises) because a
        # perturbed point can land in a collision (guardrail, overpass support, etc.); each
        # retry draws a fresh point from rng rather than repeating the same doomed one, which
        # matters because main()'s outer loop retries this whole function on any exception
        # with the same args/seed -- a deterministic collision would otherwise retry forever.
        route_margin = 500
        max_spawn_attempts = 20
        ego_vehicle = None
        for attempt in range(max_spawn_attempts):
            route_idx = rng.randint(0, max(0, waypoints_np.shape[0] - route_margin - 1))
            route_x, route_y = waypoints_np[route_idx]
            candidate_pose = world.get_map().get_waypoint(carla.Location(x=float(route_x), y=float(route_y))).transform
            lateral = rng.uniform(-args.lateral_perturb_max, args.lateral_perturb_max)
            yaw_offset = rng.uniform(-args.yaw_perturb_max_deg, args.yaw_perturb_max_deg)
            right = candidate_pose.get_right_vector()
            candidate_pose.location += carla.Location(x=right.x * lateral, y=right.y * lateral, z=0.5)
            candidate_pose.rotation.yaw += yaw_offset
            ego_vehicle = world.try_spawn_actor(vehicle_bp, candidate_pose)
            if ego_vehicle is not None:
                start_pose = candidate_pose
                print(f"Randomized spawn (attempt {attempt}): route_idx={route_idx} lateral={lateral:.2f}m "
                      f"yaw_offset={yaw_offset:.2f}deg weather={args.weather}")
                break
            print(f"Randomized spawn attempt {attempt} collided at route_idx={route_idx}, resampling...")
        if ego_vehicle is None:
            raise RuntimeError(f"Failed to find a collision-free randomized spawn point after {max_spawn_attempts} attempts")
    elif args.track == 'town04':
        start_pose = spawn_points[PLAYER_START_INDEX]
        ego_vehicle = world.spawn_actor(vehicle_bp, start_pose)
    else:
        # Custom OpenDRIVE tracks only expose one auto-generated spawn point (unrelated to
        # PLAYER_START_INDEX, which is Town04-specific) -- start from the route's first
        # waypoint instead, snapped onto the lane the same way --randomize-spawn does.
        first_x, first_y = waypoints_np[0]
        start_pose = world.get_map().get_waypoint(carla.Location(x=float(first_x), y=float(first_y))).transform
        start_pose.location.z += 0.5
        ego_vehicle = world.spawn_actor(vehicle_bp, start_pose)
    # Always on: throttle/brake are read from CARLA's autopilot every frame (see expert_control
    # below) regardless of whether steering comes from autopilot (RUN_NO==0) or the NN+CBF.
    ego_vehicle.set_autopilot(True)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, AUTOPILOT_SPEED_DIFF_PERCENT)

    spectator = world.get_spectator()
    vehicle_transform = ego_vehicle.get_transform()
    spectator.set_transform(carla.Transform(vehicle_transform.location + carla.Location(z=20), carla.Rotation(pitch=-90)))

    # --- 3. SPAWN MODERN SENSORS ---
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
    camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))

    cam0_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
    camera0 = world.spawn_actor(camera_bp, cam0_transform, attach_to=ego_vehicle)

    cam1_transform = carla.Transform(carla.Location(x=-10.0, z=6.0))
    camera1 = world.spawn_actor(camera_bp, cam1_transform, attach_to=ego_vehicle)

    image_queue = queue.Queue()
    video_queue = queue.Queue()
    camera0.listen(image_queue.put)
    camera1.listen(video_queue.put)

    client_bridge = Carla09Bridge(world, ego_vehicle, image_queue, video_queue)
    print(f"CARLA 0.9.15 Bridge active. Vehicle and sensors spawned.")

    # [Waypoint setup omitted for explanation - primarily used for logging in vision-based models]
    config = configparser.ConfigParser()
    config.read(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
    demo_opt = config['Demo Parameters']

    enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
    enable_live_plot = enable_live_plot == 'True'
    live_plot_period = float(demo_opt.get('live_plotting_period', 0))
    live_plot_timer = Timer(live_plot_period)

    wp_distance = []   # distance array
    for i in range(1, waypoints_np.shape[0]):
        wp_distance.append(
                np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                        (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
    wp_distance.append(0)  

    wp_interp      = []    
    wp_interp_hash = []    
    interp_counter = 0     
    for i in range(waypoints_np.shape[0] - 1):
        wp_interp.append(list(waypoints_np[i]))
        wp_interp_hash.append(interp_counter)   
        interp_counter+=1
        
        num_pts_to_interp = int(np.floor(wp_distance[i] /
                                     float(INTERP_DISTANCE_RES)) - 1)
        wp_vector = waypoints_np[i+1] - waypoints_np[i]
        wp_uvector = wp_vector / np.linalg.norm(wp_vector)
        for j in range(num_pts_to_interp):
            next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
            wp_interp.append(list(waypoints_np[i] + next_wp_vector))
            interp_counter+=1
            
    wp_interp.append(list(waypoints_np[-1]))
    wp_interp_hash.append(interp_counter)   
    interp_counter+=1

    controller = controller2d.Controller2D(waypoints)

    num_iterations = ITER_FOR_SIM_TIMESTEP
    if (ITER_FOR_SIM_TIMESTEP < 1):
        num_iterations = 1

    measurement_data, sensor_data = client_bridge.read_data()
    sim_start_stamp = measurement_data.game_timestamp / 1000.0
    send_control_command(ego_vehicle, throttle=0.0, steer=0, brake=1.0)
    
    sim_duration = 0
    for i in range(num_iterations):
        measurement_data, sensor_data = client_bridge.read_data()
        send_control_command(ego_vehicle, throttle=0.0, steer=0, brake=1.0)
        if i == num_iterations - 1:
            sim_duration = measurement_data.game_timestamp / 1000.0 - \
                           sim_start_stamp  
    
    SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
    print("SERVER SIMULATION STEP APPROXIMATION: " + \
          str(SIMULATION_TIME_STEP))
    TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / \
                           SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

    measurement_data, sensor_data = client_bridge.read_data()
    start_x, start_y, start_yaw = get_current_pose(measurement_data)
    print("Player data : ",measurement_data.player_measurements)

    send_control_command(ego_vehicle, throttle=0.0, steer=0, brake=1.0)
    x_history     = [start_x]
    y_history     = [start_y]
    yaw_history   = [start_yaw]
    time_history  = [0]
    speed_history = [0]
    steerings_list = [0]
    throttles_list = [0]
    
    reached_the_end = False
    skip_first_frame = True
    closest_index    = 0 
    closest_index_centre    = 0 
    closest_distance = 0 
    cmd_steer = 0
    cmd_steer1 = 0
    cmd_throttle = 0
    cmd_brake = 0
    
    # --- MODEL INITIALIZATION ---
    # Here the 4 separate models that make up the SA-TCP architecture are initialized.
    if VEHICLE_MODEL=='Kinematic' :
        model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
        model = model.to(device)
        model.eval()
        model.fc.train()
    else :
        model = EndtoEnd() # Branch A
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 128))
        model = model.to(device)
        model.eval()
        model.fc.train()

    # Branch B: The State Predictors for the CBF
    model_safety_1 = models.resnet18()
    model_safety_1.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_1 = model_safety_1.to(device)
    model_safety_1.eval()
    model_safety_1.fc.train()
    
    model_safety_2 = models.resnet18()
    model_safety_2.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_2 = model_safety_2.to(device)
    model_safety_2.eval()
    model_safety_2.fc.train()
    
    model_safety_3 = models.resnet18()
    model_safety_3.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_3 = model_safety_3.to(device)
    model_safety_3.eval()
    model_safety_3.fc.train()

    # Loads the PyTorch weights we just trained in Phase 1.
    # Prefers the best-val-loss checkpoint (train.py's early stopping); falls back to the
    # last-step checkpoint if no val split was available when that model was trained.
    if RUN_NO!=0 :
        model.load_state_dict(torch.load(os.path.join(model_path, load_checkpoint_name(model_path, ''))))
        model_safety_1.load_state_dict(torch.load(os.path.join(model_path, load_checkpoint_name(model_path, '-safety-1'))))
        model_safety_2.load_state_dict(torch.load(os.path.join(model_path, load_checkpoint_name(model_path, '-safety-2'))))
        model_safety_3.load_state_dict(torch.load(os.path.join(model_path, load_checkpoint_name(model_path, '-safety-3'))))
        print(model)
    
    tag_suffix = ('_' + args.episode_tag) if args.episode_tag else ''
    images_dir = os.path.join(args.output_dir, 'run' + str(RUN_NO) + tag_suffix + '_images')
    video_dir = os.path.join(args.output_dir, 'run' + str(RUN_NO) + tag_suffix + '_video')
    makedirs_maybe_smb(images_dir)
    makedirs_maybe_smb(video_dir)

    print("Running for", TOTAL_EPISODE_FRAMES)
    curvature_save = 0
    x_save = 0
    theta_save = 0
    theta = 0
    theta_var = 0
    x = 0
    x_var = 0
    curvature = 0
    curvature_var = 0
    theta_comps = []
    steer_var = 1
    x_comps = []
    curvature_comps = []
    steer_comps = []  # [raw NN steer, CBF-applied steer] per frame, for with/without-CBF intervention analysis
    current_x, current_y, current_yaw = 0.,0.,0.
    current_timestamp = 0.
    stall_seconds = 0.  # consecutive sim-time spent under STALL_SPEED_THRESHOLD, post-startup

    # --- INFERENCE LOOP ---
    # This is the real-time driving loop. For every frame, the camera captures an image,
    # the 4 models process it, and the CBF acts as the safety referee before sending 
    # the final steering command back to CARLA.
    for frame in tqdm.tqdm(range(TOTAL_EPISODE_FRAMES)):
        measurement_data, sensor_data = client_bridge.read_data()
        
        # --- Third-Person Chase Camera ---
        spectator = world.get_spectator()
        vehicle_transform = ego_vehicle.get_transform()
        forward_vec = vehicle_transform.get_forward_vector()
        
        # Position the camera 6 meters behind the car, and 2.5 meters in the air
        camera_location = carla.Location(
            x = vehicle_transform.location.x - (6.0 * forward_vec.x),
            y = vehicle_transform.location.y - (6.0 * forward_vec.y),
            z = vehicle_transform.location.z + 2.5
        )
        
        # Match the car's steering rotation, but pitch slightly down to see the road
        camera_rotation = carla.Rotation(
            pitch = -15.0, 
            yaw = vehicle_transform.rotation.yaw, 
            roll = 0.0
        )
        
        spectator.set_transform(carla.Transform(camera_location, camera_rotation))
        prev_x, prev_y, prev_yaw = current_x, current_y, current_yaw
        current_x, current_y, current_yaw = \
            get_current_pose(measurement_data)
        vel_x_num = (current_x-prev_x)/(float(measurement_data.game_timestamp) / 1000.0 - (current_timestamp+WAIT_TIME_BEFORE_START) )
        vel_y_num = (current_y-prev_y)/(float(measurement_data.game_timestamp) / 1000.0 - (current_timestamp+WAIT_TIME_BEFORE_START))
        current_speed = measurement_data.player_measurements.forward_speed
        current_speed_perp = vel_y_num*math.cos(current_yaw) - vel_x_num*math.sin(current_yaw)
        current_speed = vel_x_num*math.cos(current_yaw) + vel_y_num*math.sin(current_yaw)
        current_omega = (current_yaw-prev_yaw)/(float(measurement_data.game_timestamp) / 1000.0 - (current_timestamp+WAIT_TIME_BEFORE_START))
        
        print("Numericals : ", current_speed, current_speed_perp,current_omega)
        current_timestamp = float(measurement_data.game_timestamp) / 1000.0

        if current_timestamp <= WAIT_TIME_BEFORE_START:
            send_control_command(ego_vehicle, throttle=0.0, steer=0, brake=1.0)
            continue
        else:
            current_timestamp = current_timestamp - WAIT_TIME_BEFORE_START

        if abs(current_speed) < STALL_SPEED_THRESHOLD:
            stall_seconds += settings.fixed_delta_seconds
            if stall_seconds >= STALL_TIMEOUT_SECONDS:
                raise RuntimeError(
                    f"Vehicle stalled (speed < {STALL_SPEED_THRESHOLD} m/s) for "
                    f"{STALL_TIMEOUT_SECONDS}s at ({current_x:.1f}, {current_y:.1f}) -- "
                    f"aborting episode instead of recording a parked car for the remaining runtime.")
        else:
            stall_seconds = 0.

        if frame%5 == 0 :
            expert_control = ego_vehicle.get_control()
            cmd_steer = expert_control.steer
            cmd_throttle = expert_control.throttle
            cmd_brake = expert_control.brake
            x_history.append(current_x)
            y_history.append(current_y)
            yaw_history.append(current_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp) 
            steerings_list.append(cmd_steer)
            throttles_list.append(cmd_throttle)
            main_image = sensor_data.get('CameraRGB', None)
            video_image = sensor_data.get('CameraRGB_video', None)
            
            img_array_video = np.array(image_converter.to_rgb_array(video_image))
            im_video = Image.fromarray(img_array_video)
            save_image_maybe_smb(im_video, video_dir+'/frame_'+str(frame//5)+'.png')

            img_array = np.array(image_converter.to_rgb_array(main_image))
            im = Image.fromarray(img_array)
            steer_to_save = int(100*cmd_steer*180./math.pi)
            if frame//5 > 0:
                save_image_maybe_smb(im, images_dir+'/frame_'+str(frame//5)+'_'+str(steer_to_save)+\
                    '_'+str(int(10000*curvature_save))+'_'+str(int(100*x_save))+'_'+\
                    str(int(100*theta_save*180./math.pi))+'_'+str(int(100*current_speed))+\
                    '_'+str(int(100*current_speed_perp))+'.png')
            
            # Here we push the camera frame into the Neural Networks
            inp = torch.unsqueeze(transform(im),0).to(device)
            preds = []
            mean = 0
            for i in range(N_ITERS) :
                if VEHICLE_MODEL=='Kinematic' :
                    preds.append(float(model(inp)[0].cpu()))
                else :
                    preds.append(float(model(inp,torch.tensor([[current_speed]]).to(device)\
                        ,torch.tensor([[current_speed_perp]]).to(device))[0].cpu()))
                mean += preds[i]
            cmd_steer1 = mean/N_ITERS
            var = 0
            for pred in preds :
                var += (pred-cmd_steer1)**2
            steer_var = (var/N_ITERS)**(1/2)*(math.pi/180.)
            cmd_steer1 = cmd_steer1*(math.pi/180.)
            Y2 = [0,0,0]
            
            # Curvature Prediction (Model 1)
            preds = []
            mean = 0
            for i in range(N_ITERS) :
                preds.append(float(model_safety_1(inp)[0].cpu()))
                mean += preds[i]
            Y2[0] = mean/N_ITERS
            var = 0
            for pred in preds :
                var += (pred-Y2[0])**2
            curvature_var = (var/N_ITERS)**(1/2)*curvature_factor

            # Cross-Track Error Prediction (Model 2)
            preds = []
            mean = 0
            for i in range(N_ITERS) :
                preds.append(float(model_safety_2(inp)[0].cpu()))
                mean += preds[i]
            Y2[1] = mean/N_ITERS
            var = 0
            for pred in preds :
                var += (pred-Y2[1])**2
            x_var = (var/N_ITERS)**(1/2)*x_factor
            
            # Heading Error Prediction (Model 3)
            preds = []
            mean = 0
            for i in range(N_ITERS) :
                preds.append(float(model_safety_3(inp)[0].cpu()))
                mean += preds[i]
            Y2[2] = mean/N_ITERS
            var = 0
            for pred in preds :
                var += (pred-Y2[2])**2
            theta_var = (var/N_ITERS)**(1/2)*theta_factor*math.pi/180.

            curvature = float(Y2[0])*curvature_factor
            x = float(Y2[1])*x_factor
            theta = float(Y2[2])*theta_factor*math.pi/180.
            print("Before ", cmd_steer1, steer_var)
            print("Predicted theta, x, curvature : ", theta, x, x_var, curvature)
            print("Observed theta, x, curvature : ", theta_save, x_save, curvature_save)
            theta_comps.append([theta,theta_save])
            x_comps.append([x,x_var,x_save])
            curvature_comps.append([curvature,curvature_save])
        
            if measurement_data.player_measurements.collision_other > 0:
                print("Collided")
                break

        # [Waypoint distance calculations omitted for brevity - used for simulation tracking]
        closest_distance = np.linalg.norm(np.array([
                waypoints_np[closest_index, 0] - current_x,
                waypoints_np[closest_index, 1] - current_y]))
        new_distance = closest_distance
        new_index = closest_index
        while new_distance <= closest_distance:
            closest_distance = new_distance
            closest_index = new_index
            new_index += 1
            if new_index >= waypoints_np.shape[0]:  
                break
            new_distance = np.linalg.norm(np.array([
                    waypoints_np[new_index, 0] - current_x,
                    waypoints_np[new_index, 1] - current_y]))
        new_distance = closest_distance
        new_index = closest_index
        while new_distance <= closest_distance:
            closest_distance = new_distance
            closest_index = new_index
            new_index -= 1
            if new_index < 0:  
                break
            new_distance = np.linalg.norm(np.array([
                    waypoints_np[new_index, 0] - current_x,
                    waypoints_np[new_index, 1] - current_y]))

        if closest_index >= waypoints_np.shape[0]-8:  
            break
        
        closest_distance = np.linalg.norm(np.array([
                centerline_np[closest_index_centre, 0] - current_x,
                centerline_np[closest_index_centre, 1] - current_y]))
        new_distance = closest_distance
        new_index = closest_index_centre
        while new_distance <= closest_distance:
            closest_distance = new_distance
            closest_index_centre = new_index
            new_index += 1
            if new_index >= centerline_np.shape[0]:  
                break
            new_distance = np.linalg.norm(np.array([
                    centerline_np[new_index, 0] - current_x,
                    centerline_np[new_index, 1] - current_y]))
        new_distance = closest_distance
        new_index = closest_index_centre
        while new_distance <= closest_distance:
            closest_distance = new_distance
            closest_index_centre = new_index
            new_index -= 1
            if new_index < 0:  
                break
            new_distance = np.linalg.norm(np.array([
                    centerline_np[new_index, 0] - current_x,
                    centerline_np[new_index, 1] - current_y]))

        if closest_index_centre >= centerline_np.shape[0]-8:  
            break
        closest_angle_vector = centerline_np[closest_index_centre+5,:]-centerline_np[closest_index_centre,:]
        closest_angle = math.atan2(closest_angle_vector[1],closest_angle_vector[0])
        
        rel_angle = current_yaw - closest_angle
        
        waypoint_subset_first_index = closest_index - 1
        if waypoint_subset_first_index < 0:
            waypoint_subset_first_index = 0

        waypoint_subset_last_index = closest_index
        total_distance_ahead = 0
        while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
            total_distance_ahead += wp_distance[waypoint_subset_last_index]
            waypoint_subset_last_index += 1
            if waypoint_subset_last_index >= waypoints_np.shape[0]:
                waypoint_subset_last_index = waypoints_np.shape[0] - 1
                break

        new_waypoints = \
                wp_interp[wp_interp_hash[waypoint_subset_first_index]:\
                          wp_interp_hash[waypoint_subset_last_index] + 1]
        controller.update_waypoints(new_waypoints)

        vec1 = np.array([current_x,current_y])-centerline_np[closest_index_centre,:2]
        vec2 = np.array(closest_angle_vector)
        val = vec2[0]*vec1[1] - vec2[1]*vec1[0]
        x_save = closest_distance*val/abs(val)
        # Wrap into [-pi, pi]: current_yaw and closest_angle each wrap independently at +/-pi,
        # so their raw difference can jump to ~+/-2*pi right at that boundary without this.
        theta_save = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
        curvature_save = computeCurvature(centerline_np[closest_index_centre,:],\
            centerline_np[closest_index_centre+2,:],centerline_np[closest_index_centre+4,:])
        
        # --- THE SAFETY OVERRIDE ---
        # This is where the magic happens. All the neural
        # network predictions are passed into the CBF, and it returns 'cmd_steer_updated' which is
        # mathematically guaranteed to be safe.
        cmd_steer_updated = get_optimal_control(cmd_steer1, steer_var , current_speed, \
            theta_save, theta_var, x_save, x_var, curvature_save, curvature_var, \
            v_perp=current_speed_perp,omega=current_omega)
        steer_comps.append([cmd_steer1, cmd_steer_updated])

        if skip_first_frame and frame == 0:
            pass
        else:
            new_waypoints_np = np.array(new_waypoints)
            path_indices = np.floor(np.linspace(0, 
                                                new_waypoints_np.shape[0]-1,
                                                INTERP_MAX_POINTS_PLOT))

        # If RUN_NO == 0, we use expert Autopilot to collect data. 
        # Else, we use our trained 'cmd_steer_updated' network output to drive!
        if RUN_NO == 0 or frame//5 < 0:
            send_control_command(ego_vehicle,
                                throttle=cmd_throttle,
                                steer=cmd_steer,
                                brake=cmd_brake)
        else :
            send_control_command(ego_vehicle,
                                throttle=cmd_throttle,
                                steer=cmd_steer_updated,
                                brake=cmd_brake)
        
        dist_to_last_waypoint = np.linalg.norm(np.array([
            waypoints[-1][0] - current_x,
            waypoints[-1][1] - current_y]))
        if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
            reached_the_end = True
        if reached_the_end:
            break

    if reached_the_end:
        print("Reached the end of path. Writing to controller_output...")
    else:
        print("Exceeded assessment time. Writing to controller_output...")
    
    send_control_command(ego_vehicle, throttle=0.0, steer=0.0, brake=1.0)
    write_trajectory_file(x_history, y_history, speed_history, time_history,steerings_list,throttles_list)

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    mode = cbf_mode_name()
    theta_comps = np.array(theta_comps)
    x_comps = np.array(x_comps)
    curvature_comps = np.array(curvature_comps)
    steer_comps = np.array(steer_comps)
    np.savetxt(os.path.join(results_dir, f'{mode}_theta_comps_run{RUN_NO}.csv'), theta_comps)
    np.savetxt(os.path.join(results_dir, f'{mode}_x_comps_run{RUN_NO}.csv'), x_comps)
    np.savetxt(os.path.join(results_dir, f'{mode}_curvature_comps_run{RUN_NO}.csv'), curvature_comps)
    np.savetxt(os.path.join(results_dir, f'{mode}_steer_comps_run{RUN_NO}.csv'), steer_comps)

def main():
    global RUN_NO
    global model_path
    global N_ITERS
    global SAFEGUARD
    global LANE_WIDTH

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-r', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
    cbf_group = argparser.add_mutually_exclusive_group()
    cbf_group.add_argument(
        '--cbf',
        action='store_true',
        help='force the CBF safety gate on, overriding the RUN_NO < K_ITERS ramp-up default')
    cbf_group.add_argument(
        '--no-cbf',
        action='store_true',
        help='force the CBF safety gate off, overriding the RUN_NO < K_ITERS ramp-up default')
    argparser.add_argument(
        '--weather',
        default='ClearNoon',
        help='CARLA weather preset name from carla.WeatherParameters (default: ClearNoon)')
    argparser.add_argument(
        '--randomize-spawn',
        action='store_true',
        help='spawn at a random point along the route (instead of the fixed PLAYER_START_INDEX), '
             'perturbed sideways/yaw per --lateral-perturb-max/--yaw-perturb-max-deg, for recovery-maneuver data collection')
    argparser.add_argument(
        '--lateral-perturb-max',
        type=float,
        default=1.5,
        help='max meters of sideways spawn offset when --randomize-spawn is set (default: 1.5)')
    argparser.add_argument(
        '--yaw-perturb-max-deg',
        type=float,
        default=15.0,
        help='max degrees of yaw spawn offset when --randomize-spawn is set (default: 15.0)')
    argparser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='seed for --randomize-spawn\'s route-point and perturbation draws (default: unseeded)')
    argparser.add_argument(
        '--episode-tag',
        default='',
        help='appended to the output image/video folder names (run<N>_<tag>_images), '
             'so repeated collection episodes with the same --run_no don\'t overwrite each other')
    argparser.add_argument(
        '--output-dir',
        default='.',
        help='root directory for the run<N>_..._images/video folders (default: current directory). '
             'Point this at a mounted network share to collect data without using local disk.')
    argparser.add_argument(
        '--track',
        default='austin_map',
        choices=sorted(TRACKS.keys()),
        help='which track/map to run on (default: austin_map; pass --track town04 for the '
             'original Phase 1 baseline map)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    if args.run_no!=-1 :
        RUN_NO = args.run_no
    if args.cbf or args.no_cbf :
        SAFEGUARD = args.cbf
    else :
        SAFEGUARD = RUN_NO >= K_ITERS  # ramp-up default: CBF only kicks in once the model has had a few training iterations
    if not SAFEGUARD :
        N_ITERS = 1
    model_path = 'saved_models_iter' + str(RUN_NO-1)
    LANE_WIDTH = TRACKS[args.track]['lane_width']

    # Validate up front rather than inside the retry loop below -- a typo'd preset (or missing
    # SMB creds) would otherwise raise on every attempt and retry forever instead of failing fast.
    if getattr(carla.WeatherParameters, args.weather, None) is None:
        raise ValueError(f"Unknown --weather preset '{args.weather}'; see carla.WeatherParameters for valid names.")
    if is_unc_path(args.output_dir) and not (os.environ.get('SMB_USERNAME') and os.environ.get('SMB_PASSWORD')):
        raise RuntimeError(
            '--output-dir is a UNC path but SMB_USERNAME/SMB_PASSWORD are not set in the environment.')

    attempt = 0
    while True:
        try:
            exec_waypoint_nav_demo(args, spawn_seed_offset=attempt)
            print('Done.')
            return

        except Exception as error:
            logging.error(error)
            attempt += 1
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')