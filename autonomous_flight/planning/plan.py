# Imports
# ==================================================

import argparse
import time
import msgpack
import visdom
import config
import numpy as np

from udacidrone             import Drone
from udacidrone.connection  import MavlinkConnection
from udacidrone.messaging   import MsgID
from udacidrone.frame_utils import global_to_local


from utils import (
    a_star, 
    gen_plots,
    heuristic, 
    create_grid, 
    prune_path,
    States
)



# Drone Class
# ==================================================

class MotionPlanning(Drone):
    def __init__(self, connection, home_lon, home_lat, g_position, use_visdom):
        super().__init__(connection)


        # Initialize Drone
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints       = []
        self.in_mission      = True
        self.check_state     = {}
        self.flight_state    = States.MANUAL
        self.g_position      = g_position
        self.home_lon        = home_lon
        self.home_lat        = home_lat
        self.use_visdom      = use_visdom
        self.start_time      = time.time()


        # Register Callbacks
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE,          self.state_callback)
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)


        # Initialize Visdom
        if self.use_visdom == 'true':
            self.v = visdom.Visdom()

            self.path = {
                'seconds' : [],
                'north'   : [],
                'east'    : [],
                'down'    : []
            }

            self.velocity = {
                'seconds' : [],
                'north'   : [],
                'east'    : [],
                'down'    : []
            } 

            if self.v.check_connection():
                print(f"[INFO] visdom available - {config.VISDOM_LINK}")

                self.register_callback(MsgID.LOCAL_POSITION, self.record_stats_callback)


    # Callbacks
    # --------------------------------------------------
    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > config.DEADBAND_ALTITUDE * self.target_position[2]:
                self.waypoint_transition()

        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < config.DEADBAND_RADIUS:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()

                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()


    def record_stats_callback(self):
        """
        """

        seconds = round(time.time()) - round(self.start_time)

        # Record Position
        self.path['seconds'].append(seconds)
        self.path['north'  ].append( self.local_position[0])
        self.path['east'   ].append( self.local_position[1])
        self.path['down'   ].append(-self.local_position[2])


        # Record Velocity
        self.velocity['seconds'].append(seconds)
        self.velocity['north'  ].append(self.local_velocity[0])
        self.velocity['east'   ].append(self.local_velocity[1])
        self.velocity['down'   ].append(self.local_velocity[2])


    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()

            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()

            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()

            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()


    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()


    # Transitions
    # --------------------------------------------------
    def arming_transition(self):
        """
        """
        
        print("[INFO] arming transition")

        self.arm()
        self.take_control()

        self.flight_state = States.ARMING


    def disarming_transition(self):
        """
        """

        print("[INFO] disarm transition")

        self.disarm()
        self.release_control()

        self.flight_state = States.DISARMING


    def landing_transition(self):
        """
        """

        print("[INFO] landing transition")

        self.land()

        self.flight_state = States.LANDING


    def manual_transition(self):
        """
        """

        print("[INFO] manual transition")

        self.stop()
        self.in_mission = False

        self.flight_state = States.MANUAL


    def takeoff_transition(self):
        """
        """

        print("[INFO] takeoff transition")

        self.takeoff(self.target_position[2])

        self.flight_state = States.TAKEOFF


    def waypoint_transition(self):
        """
        """

        self.target_position = self.waypoints.pop(0)

        print(f"[INFO] target position: {self.target_position}")

        self.cmd_position(
            self.target_position[0],
            self.target_position[1],
            self.target_position[2],
            self.target_position[3]
        )

        self.flight_state = States.WAYPOINT


    # Actions
    # --------------------------------------------------
    def plan_path(self):
        """
        """

        print("[INFO] searching for path")


        # Load Map
        data= np.loadtxt(
            fname     = config.CITY_MAP, 
            delimiter = config.CSV_DELIM, 
            dtype     = 'Float64', 
            skiprows  = 2
        )

        grid, north_offset, east_offset = create_grid(data, config.TARGET_ALTITUDE, config.SAFETY_DISTANCE)

        print(
            f"[INFO] offset meters north: {north_offset}\n"
            f"       offset meters east:  {east_offset}"
        )

        
        # Set Home Position
        self.set_home_position(
            longitude = self.home_lon,
            latitude  = self.home_lat,
            altitude  = self.target_position[2]
        )

        self.target_position[2] = config.TARGET_ALTITUDE

        print(
            f"[INFO] global home:     {np.array(self.global_home    ).round(5)}\n"
            f"       global position: {np.array(self.global_position).round(5)}\n"
            f"       local position:  {np.array(self.local_position ).round(5)}"
        )


        # Set Start Position
        start_coordinates_NED = global_to_local(
            global_position = [
                self._longitude, 
                self._latitude, 
                self._altitude
            ], 
            global_home = self.global_home
        )

        grid_start = (
            int(start_coordinates_NED[0] - north_offset),
            int(start_coordinates_NED[1] - east_offset)
        )


        # Set Goal Position
        goal_coordinates_NED = global_to_local(
            global_position = [
                self.g_position[0], 
                self.g_position[1], 
                self._altitude
            ], 
            global_home = self.global_home
        )


        # Check if Goal is Feasible
        if int(goal_coordinates_NED[0] - north_offset) in range(0, grid.shape[0] - 1) and \
           int(goal_coordinates_NED[1] -  east_offset) in range(0, grid.shape[1] - 1):
            grid_goal = (
                int(goal_coordinates_NED[0] - north_offset),
                int(goal_coordinates_NED[1] - east_offset)
            )


        # Otherwise Use Default Plan
        else:
            print(
                f"[WARN] goal position outside map range\n"
                f"       max range north: {grid.shape[0]}\n"
                f"       max range east:  {grid.shape[1]}\n"
                f"\n"
                f"       goal position north: {int(goal_coordinates_NED[0] - north_offset)}\n"
                f"       goal position east:  {int(goal_coordinates_NED[1] - north_offset)}\n"
            )

            print(f"[INFO] setting default goal location")

            grid_goal = (
                int(start_coordinates_NED[0] - north_offset + config.DEFAULT_NORTH),
                int(start_coordinates_NED[1] - east_offset  + config.DEFAULT_EAST)
            )

        print(
            f"[INFO] Start: {grid_start}\n"
            f"       Goal:  {grid_goal}"
        )


        # Plan Path
        print("[INFO] planning path")

        path, _ = a_star(
            grid  = grid, 
            h     = heuristic, 
            start = grid_start, 
            goal  = grid_goal
        )


        # If Feasible Path Prune Path
        if path:
            print("[INFO] prunning path")

            path = prune_path(grid, path)


            # Setting Waypoints
            print("[INFO] sending waypoints")

            waypoints      = [[p[0] + north_offset, p[1] + east_offset, config.TARGET_ALTITUDE, 0] for p in path]
            self.waypoints = waypoints

            self.connection._master.write(msgpack.dumps(self.waypoints))

            self.flight_state = States.PLANNING


        # Otherwise Terminate Mission
        else:
            print(f"[WARN] infeasible goal designated")
            self.disarming_transition()


    def start(self):
        """
        """
        
        # Initialize Log File
        print("[INFO] ceating log file")
        self.start_log(config.LOG_DIR, config.LOG_FILE)


        # Connect to Drone
        print("[INFO] starting connection")
        self.connection.start()


        # Visdom Path
        if self.use_visdom == 'true':
            gen_plots(self.velocity, self.path, self.v)


        # Finish Plan
        print("[INFO] closing log file")
        self.stop_log()



# Application
# ==================================================
if __name__ == "__main__":
    """
    """
    
    # Get Start Lat and Lon
    with open(config.CITY_MAP) as f:
        lat, lon = [float(x.split(" ")[1]) for x in f.readline().split(f"{config.CSV_DELIM} ")]


    # Parse Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--port',       type = int,   default = config.CONN_PORT,     help = "port number")
    parser.add_argument('--host',       type = str,   default = config.CONN_HOST,     help = "host address")
    parser.add_argument('--m_north',    type = float, default = config.DEFAULT_NORTH, help = "goal meters north")
    parser.add_argument('--m_east',     type = float, default = config.DEFAULT_EAST,  help = "goal meters east")
    parser.add_argument('--use_visdom', type = str,   default = config.USE_VISDOM,    help = "visdom if server running")

    args  = parser.parse_args()


    # Convert Goal to Lat/Lon
    goal_lat = lat + (args.m_north  * config.METERS_2_LAT)
    goal_lon = lon + (args.m_east   * config.METERS_2_LON)


    # Create Drone Connection
    conn  = MavlinkConnection(f"tcp:{args.host}:{args.port}", timeout = config.CONN_TIMEOUT)
    drone = MotionPlanning(
        connection = conn,
        home_lon   = lon,
        home_lat   = lat,
        g_position = [goal_lon, goal_lat],
        use_visdom = args.use_visdom
    )

    time.sleep(1)

    drone.start()