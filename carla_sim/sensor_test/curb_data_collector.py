import math
import os.path
import sys
import argparse
import random
import time
import carla
from queue import Queue
import numpy as np

import pygame  # for visualize cameras
from pygame.locals import K_ESCAPE, K_q
import open3d as o3d  # for visualize LiDAR and Radar
from matplotlib import cm

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL_RANGE = np.linspace(0, 1, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:, :3]
global global_sensor_number
global_sensor_number = 0


def parser():
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
        default=10,
        type=int,
        help='number of vehicles (default: 30)')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        default=True,
        help='default is False, ie, asynchronous mode')
    argparser.add_argument(
        '--dormant_distance',
        default=2000,
        type=int,
        help='dormant distance (default: 2000)'
    )
    argparser.add_argument(
        '--seed',
        default=time.time(),
        type=int,
        help='random seed (default: time.time())'
    )
    argparser.add_argument(
        '--store-data',
        action='store_true',
        default=False,
        help='store data to disk (default: False)'
    )
    return argparser.parse_args()


class DisplayManager(object):
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_display_offset(self, grid_pos):
        dis_size = self.get_display_size()
        return [int(grid_pos[1] * dis_size[0]), int(grid_pos[0] * dis_size[1])]

    def render_enabled(self):
        return self.display != None

    def render(self):
        if not self.render_enabled():
            return

        for sensor in self.sensor_list:
            sensor.render()
        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()


class SensorManager(object):
    def __init__(self, world, display_manager, sensor_type, transform, attached, sensor_options, display_pos, store,
                 sensor_name, point_list=None):
        '''

        :param world: carla world object
        :param display_manager: display manager object
        :param sensor_type: sensor type
        :param transform: sensor relative location
        :param attached: attached object
        :param sensor_options: sensor options
        :param display_pos: display position
        :param store: store to disk
        :param sensor_name:
        :param point_list:
        '''
        global global_sensor_number
        self.surface = None
        self.world = world
        if "LiDAR" in sensor_type or 'Radar' in sensor_type:
            self.display_manager = None
        else:
            self.display_manager = display_manager
            self.display_manager.add_sensor(self)
            self.display_pos = display_pos

        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options, point_list)
        self.sensor_option = sensor_options
        self.store = store
        self.sensor_name = sensor_name
        global_sensor_number += 1

    def init_sensor(self, sensor_type, transform, attached, sensor_options, point_list=None):
        if sensor_type == 'RGBCamera':
            print('Initializing RGB sensor...')
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_manager.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options.keys():
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)
            return camera

        elif 'LiDAR' in sensor_type:
            print('Initializing Lidar sensor...')
            if sensor_type == 'SemanticLiDAR':
                lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            else:
                lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
                lidar_bp.set_attribute('dropoff_general_rate',
                                       lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
                lidar_bp.set_attribute('dropoff_intensity_limit',
                                       lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
                lidar_bp.set_attribute('dropoff_zero_intensity',
                                       lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            if sensor_type == 'SemanticLiDAR':
                pass
            else:
                lidar.listen(lambda data: self.save_lidar_image(data, point_list))

            return lidar

        elif sensor_type == 'Radar':
            print('Initializing Radar sensor...')
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])
            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(lambda data: self.save_radar_image(data, point_list))
            return radar

    def save_rgb_image(self, image):
        # Raw converter means no changes applied to this image
        image.convert(carla.ColorConverter.Raw)
        image_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        image_array = np.reshape(image_array, (image.height, image.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]
        if self.store:
            image.save_to_disk(os.path.join('output', '%6d_%s.png' % (image.frame, self.sensor_name)), )
            sensor_queue.put((image.frame, self.sensor_name))

        if self.display_manager.render_enabled():
            self.surface = pygame.surfarray.make_surface(image_array.swapaxes(0, 1))

    def save_lidar_image(self, point_cloud, point_list):
        """Prepares a point cloud with intensity
        colors ready to be consumed by Open3D"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        if self.store:
            np.save('./output/%6d_%s.npy' % (point_cloud.frame, self.sensor_name), data)
            sensor_queue.put((point_cloud.frame, self.sensor_name))

        # Isolate the intensity and compute a color for it
        intensity = data[:, -1]
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        # Isolate the 3D data
        points = data[:, :-1]

        # We're negating the y to correclty visualize a world that matches
        # what we see in Unreal since Open3D uses a right-handed coordinate system
        points[:, :1] = -points[:, :1]

        # # An example of converting points from sensor to vehicle space if we had
        # # a carla.Transform variable named "tran":
        # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        # points = np.dot(tran.get_matrix(), points.T).T
        # points = points[:, :-1]

        point_list.points = o3d.utility.Vector3dVector(points)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

    def save_radar_image(self, point_cloud, point_list):

        if self.store:
            save_points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
            save_points = np.reshape(save_points, (len(point_cloud), 4))
            np.save('./output/%6d_%s.npy' % (point_cloud.frame, self.sensor_name), save_points)
            sensor_queue.put((point_cloud.frame, self.sensor_name))

        radar_data = np.zeros((len(point_cloud), 4))
        for i, detection in enumerate(point_cloud):
            x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
            y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
            z = detection.depth * math.sin(detection.altitude)
            radar_data[i, :] = [x, y, z, detection.velocity]

        intensity = np.abs(radar_data[:, -1])
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
            np.interp(intensity_col, COOL_RANGE, COOL[:, 2])
        ]

        points = radar_data[:, :-1]
        points_copy = points.copy()  # Create a copy of the points data
        points_copy[:, :1] = -points_copy[:, :1]  # Flip the x-axis values in the copy
        point_list.points = o3d.utility.Vector3dVector(points_copy)
        point_list.colors = o3d.utility.Vector3dVector(int_color)

    def render(self):
        if self.surface is not None:
            offset = self.display_manager.get_display_offset(self.display_pos)
            self.display_manager.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def main():
    args = parser()
    actor_list = []
    random.seed(args.seed if args.seed is not None else int(time.time()))
    if not os.path.isdir('output'):
        os.mkdir('output')

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        # traffic manager
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        # the default hybrid physics mode is 50 meters
        traffic_manager.set_respawn_dormant_vehicles(True)
        traffic_manager.global_percentage_speed_difference(30)

        world = client.get_world()
        origin_settings = world.get_settings()

        settings = world.get_settings()
        # for large maps
        settings.actor_activate_distance = args.dormant_distance
        settings.tile_stream_distance = args.dormant_distance

        # set synchronous mode
        settings.synchronous_mode = args.sync
        traffic_manager.set_synchronous_mode(args.sync)
        if args.sync:
            # 20 hz
            # synchronous mode and fixed time
            settings.fixed_delta_seconds = 0.05
        else:
            # asynchronous mode and variable time
            settings.fixed_delta_seconds = None

        world.apply_settings(settings)

        blueprints_vehicle = world.get_blueprint_library().filter("vehicle.*")
        blueprints_vehicle = sorted(blueprints_vehicle, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles >= number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            sys.stderr.write(msg % (args.number_of_vehicles, number_of_spawn_points))
            sys.stderr.flush()
            args.number_of_vehicles = number_of_spawn_points - 1

        # Use command to apply actions on batch of data
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        # this is equal to int 0
        FutureActor = carla.command.FutureActor

        batch = []

        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break

            blueprint = random.choice(blueprints_vehicle)

            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            # set autopilot
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot all together
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, False):
            if response.error:
                print(response.error)
            else:
                actor_list.append(response.actor_id)

        if not args.sync:
            world.wait_for_tick()
        else:
            world.tick()

        # all_vehicle_actors = world.get_actors(actor_list)
        # for actor in all_vehicle_actors:
        #     traffic_manager.update_vehicle_lights(actor, True)
        # print(world.get_blueprint_library())
        ego_vehicle_bp = world.get_blueprint_library().find('vehicle.bus.bus_41')
        # ego_vehicle_bp.set_attribute('color', '0, 255, 0')
        ego_vehicle_bp.set_attribute('role_name', 'hero')
        transform = spawn_points[len(actor_list)]

        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
        ego_vehicle.set_autopilot(True, args.tm_port)
        actor_list.append(ego_vehicle.id)

        print('spawned %d vehicles , press Ctrl+C to exit.' % (len(actor_list)))
        # create sensor queue

        if args.store_data:
            global sensor_queue
            sensor_queue = Queue(maxsize=100)

        display_manager = DisplayManager([1, 4], window_size=[1920, 540])

        # Camera
        SensorManager(world, display_manager, 'RGBCamera',
                      carla.Transform(carla.Location(x=5.7, z=3.2, y=-1), carla.Rotation(yaw=-60)),
                      ego_vehicle, {}, display_pos=[0, 0], store=args.store_data,
                      sensor_name='ego_vehicle_left_camera')
        SensorManager(world, display_manager, 'RGBCamera',
                      carla.Transform(carla.Location(x=5.7, z=3.2), carla.Rotation(yaw=+00)),
                      ego_vehicle, {}, display_pos=[0, 1], store=args.store_data,
                      sensor_name='ego_vehicle_front_camera')
        SensorManager(world, display_manager, 'RGBCamera',
                      carla.Transform(carla.Location(x=5.7, z=3.2, y=1), carla.Rotation(yaw=+60)),
                      ego_vehicle, {}, display_pos=[0, 2], store=args.store_data,
                      sensor_name='ego_vehicle_right_camera')
        SensorManager(world, display_manager, 'RGBCamera',
                      carla.Transform(carla.Location(x=-5.7, z=3.2), carla.Rotation(yaw=180)),
                      ego_vehicle, {}, display_pos=[0, 3], store=args.store_data,
                      sensor_name='ego_vehicle_back_camera')

        point_list = o3d.geometry.PointCloud()
        radar_list = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='BUS ADAS',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=5.7, y=1, z=1)),
                      ego_vehicle,
                      {'channels': '64', 'range': '100', 'points_per_second': '250000', 'rotation_frequency': '20'},
                      display_pos=[1, 0], store=args.store_data, sensor_name='LiDAR_1', point_list=point_list)

        SensorManager(world, display_manager, 'Radar',
                      carla.Transform(carla.Location(x=5.7, z=3.2)), ego_vehicle,
                      {'horizontal_fov': '30', 'points_per_second': '15000'},
                      display_pos=[1, 0], store=args.store_data, sensor_name='Radar', point_list=radar_list)

        # simulator loop
        call_exit = False
        frame = 0
        while True:
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        call_exit = True
                        break
            if call_exit:
                break
            # retrieve images
            if args.store_data:
                for sensor_idx in range(global_sensor_number):
                    sensor_info = sensor_queue.get(True, 1.0)
                    sys.stdout.write('Frame: %d, Sensor: %s' % (sensor_info[0], sensor_info[1]))
                    sys.stdout.flush()

            if vis:
                if frame == 2:
                    vis.add_geometry(point_list)
                    vis.add_geometry(radar_list)
                vis.update_geometry(point_list)
                vis.update_geometry(radar_list)
                # print('radar_list', radar_list)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.005)
            frame += 1

    finally:
        world.apply_settings(origin_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('destroying sensors')
        if display_manager:
            display_manager.destroy()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write('exited by user\n')
        sys.stdout.flush()
