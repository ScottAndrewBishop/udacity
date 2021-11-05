import argparse
import time
import msgpack
import numpy as np

from enum                   import Enum, auto
from utils                  import a_star, heuristic, create_grid, prune_path
from udacidrone             import Drone
from udacidrone.connection  import MavlinkConnection
from udacidrone.messaging   import MsgID
from udacidrone.frame_utils import global_to_local



class States(Enum):
    MANUAL    = auto()
    ARMING    = auto()
    TAKEOFF   = auto()
    WAYPOINT  = auto()
    LANDING   = auto()
    DISARMING = auto()
    PLANNING  = auto()



class MotionPlanning(Drone):
    def __init__(self, connection, home_lon, home_lat, g_position = None):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints       = []
        self.in_mission      = True
        self.check_state     = {}
        self.flight_state    = States.MANUAL
        self.g_position      = g_position
        self.home_lon        = home_lon
        self.home_lat        = home_lat

        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE,          self.state_callback)


    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()

        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()

                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()


    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()


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


    def arming_transition(self):
        self.flight_state = States.ARMING

        print("[INFO] arming transition")

        self.arm()
        self.take_control()


    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF

        print("[INFO] takeoff transition")

        self.takeoff(self.target_position[2])


    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT

        print("[INFO] waypoint transition")

        self.target_position = self.waypoints.pop(0)

        print('[INFO] target position', self.target_position)

        self.cmd_position(
            self.target_position[0],
            self.target_position[1],
            self.target_position[2],
            self.target_position[3]
        )


    def landing_transition(self):
        self.flight_state = States.LANDING

        print("[INFO] landing transition")

        self.land()


    def disarming_transition(self):
        self.flight_state = States.DISARMING

        print("[INFO] disarm transition")

        self.disarm()
        self.release_control()


    def manual_transition(self):
        self.flight_state = States.MANUAL

        print("[INFO] manual transition")

        self.stop()

        self.in_mission = False


    def send_waypoints(self):
        print("[INFO] sending waypoints to simulator ...")

        data = msgpack.dumps(self.waypoints)

        self.connection._master.write(data)


    def plan_path(self):
        self.flight_state = States.PLANNING

        print("[INFO] searching for a path ...")

        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        self.set_home_position(
            longitude = self.home_lon,
            latitude  = self.home_lat,
            altitude  = 0
        )

        print(
            f'[INFO] Global Home:     {self.global_home}\n'
            f'       Global Position: {self.global_position}\n'
            f'       Local position:  {self.local_position}'
        )

        data                            = np.loadtxt('colliders.csv', delimiter = ',', dtype = 'Float64', skiprows = 2)
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        print(
            f"[INFO] North Offset: {north_offset}\n"
            f"       East Offset:  {east_offset}"
        )

        start_coordinates_NED = global_to_local([self._longitude, self._latitude, self._altitude], self.global_home)
        grid_start            = (
            int(start_coordinates_NED[0] - north_offset),
            int(start_coordinates_NED[1] - east_offset)
        )

        goal_coordinates_NED = global_to_local([self.g_position[0], self.g_position[1], self._altitude], self.global_home)

        if int(goal_coordinates_NED[0] - north_offset) in range(0, grid.shape[0] - 1) and int(goal_coordinates_NED[1] - east_offset) in range(0, grid.shape[1] - 1):
            grid_goal = (
                int(goal_coordinates_NED[0] - north_offset),
                int(goal_coordinates_NED[1] - east_offset)
            )

        else:
            print(
                f"[WARN] goal position outside map range\n"
                f"       Max Range N: {grid.shape[0]}\n"
                f"       Max Range E: {grid.shape[1]}\n"
                f"\n"
                f"       Goal Position N: {int(goal_coordinates_NED[0] - north_offset)}\n"
                f"       Goal Position E: {int(goal_coordinates_NED[1] - north_offset)}\n"
            )

            print(f"[INFO] setting a default goal location")

            grid_goal = (
                int(start_coordinates_NED[0] - north_offset + 10),
                int(start_coordinates_NED[1] - east_offset  + 10)
            )

        print(
            f"[INFO] Start: {grid_start}\n"
            f"       Goal:  {grid_goal}"
        )

        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        path    = prune_path(path)

        waypoints      = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        self.waypoints = waypoints

        self.send_waypoints()


    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("[INFO] starting connection")
        self.connection.start()

        self.stop_log()



if __name__ == "__main__":
    with open("colliders.csv") as f:
        lat0, lon0 = [float(x.split(" ")[1]) for x in f.readline().split(", ")]

    parser = argparse.ArgumentParser()

    parser.add_argument('--port',     type = int,   default = 5760,          help = 'Port number')
    parser.add_argument('--host',     type = str,   default = '127.0.0.1',   help = "host address, i.e. '127.0.0.1'")
    parser.add_argument('--goal_lon', type = float, default = lon0 + 0.5e-5, help = "Goal Lon")
    parser.add_argument('--goal_lat', type = float, default = lat0 + 0.5e-5, help = "Goal Lat")

    args  = parser.parse_args()
    conn  = MavlinkConnection(f"tcp:{args.host}:{args.port}", timeout = 60)
    drone = MotionPlanning(
        connection = conn,
        home_lon   = lon0,
        home_lat   = lat0,
        g_position = [args.goal_lon, args.goal_lat]
    )

    time.sleep(1)

    drone.start()