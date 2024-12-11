#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time

import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random


# import pandas as pd


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        start_time = time.time()
        tls = {}  # {landmark_id: traffic_ligth_actor}

        for landmark in world.get_actors().filter('*traffic_light*'):
            x, y, z = landmark.get_location().x, landmark.get_location().y, landmark.get_location().z
            x, y, z = np.around(x, 1), np.around(y, 1), np.around(z, 1)
            if (x, y, z) in tl_dict:
                tls[(x, y, z)] = landmark

        while True:
            if not args.asynch and synchronous_master:
                world.tick()
                t = time.time() - start_time
                t_idx = t % cycle_time
                phase_name = phase_list[int(t_idx)]
                phase_green_line = phase_4_dict[phase_name]
                for tl_k, tl_v in tls.items():
                    if tl_k in phase_green_line:
                        tl_v.set_state(carla.TrafficLightState.Green)
                    else:
                        tl_v.set_state(carla.TrafficLightState.Red)
            else:
                world.wait_for_tick()

    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        time.sleep(0.5)


if __name__ == '__main__':
    # tl_pd = pd.read_csv(r'tl.csv', header=0)
    # tl_dict = dict(zip([(x, y, z) for (x, y, z) in zip(tl_pd['x'], tl_pd['y'], tl_pd['z'])], tl_pd['name']))
    tl_dict = {
        (944.6, -272.2, 70.4): 'N_1',
        (941.9, -269.7, 70.4): 'N_2',
        (918.8, -250.5, 70.7): 'W_1',
        (920.0, -247.1, 70.7): 'W_2',
        (932.3, -229.9, 71.2): 'S_1',
        (935.9, -230.5, 71.2): 'S_2',
        (963.5, -248.9, 71.0): 'E_1',
        (962.7, -252.5, 71.0): 'E_2'}

    phase_4_dict = {'NS_1': [(944.6, -272.2, 70.4), (932.3, -229.9, 71.2)],
                    'NS_2': [(941.9, -269.7, 70.4), (935.9, -230.5, 71.2)],
                    'WE_1': [(918.8, -250.5, 70.7), (963.5, -248.9, 71.0)],
                    'WE_2': [(920.0, -247.1, 70.7), (962.7, -252.5, 71.0)]}

    phase_duration_time = [4, 2, 4, 2]
    cycle_time = np.sum(phase_duration_time)
    phase_list = []
    for p_d_i, p_d_t in enumerate(phase_duration_time):
        phase_list.extend([list(phase_4_dict.keys())[p_d_i]] * p_d_t)

    print(phase_list)

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
