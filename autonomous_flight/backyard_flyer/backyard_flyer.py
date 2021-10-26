# Imports 
# ==================================================

import argparse
import time
import visdom
import numpy as np

from enum                  import Enum
from udacidrone            import Drone
from udacidrone.messaging  import MsgID
from udacidrone.connection import MavlinkConnection 



# Classes 
# ==================================================

class States(Enum):
    MANUAL    = 0
    ARMING    = 1
    TAKEOFF   = 2
    WAYPOINT  = 3
    LANDING   = 4
    DISARMING = 5


class BackyardFlyer(Drone):
    def __init__(self, connection, use_visdom, distance):
        super().__init__(connection)

        self.flight_state    = States.MANUAL
        self.start_position  = self.local_position[:2]
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.in_mission      = True
        self.all_waypoints   = []
        self.use_visdom      = use_visdom
        self.distance        = distance
        self.start_time      = time.time()

        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE,          self.state_callback)

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
                print(f"[INFO] visdom visualizations available - http://localhost:8097")

                self.register_callback(MsgID.LOCAL_POSITION, self.record_stats_callback)


    def local_position_callback(self):
        """
        """

        if self.flight_state == States.TAKEOFF:
            if -(self.local_position[2]) > 0.95 * self.target_position[2]:
                self.waypoint_transition()

        if self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[:2] - self.local_position[:2]) < 1.0 and np.linalg.norm(self.local_velocity[:2]) < 1.0:
                if self.all_waypoints: self.waypoint_transition()
                else:                  self.landing_transition()


    def velocity_callback(self):
        """
        """

        if self.flight_state == States.LANDING:
            if (self.global_position[2] - self.global_home[2] < 0.1) and abs(self.local_position[2]) < 0.01:
                self.disarming_transition()


    def state_callback(self):
        """
        """

        if not self.in_mission:
            return

        if self.flight_state == States.MANUAL:
            self.arming_transition()

        elif self.flight_state == States.ARMING:
            if self.armed:
                self.calculate_box()
                self.takeoff_transition()

        elif self.flight_state == States.DISARMING:
            if not self.armed and not self.guided:
                self.manual_transition()


    def record_stats_callback(self):
        """
        """

        seconds = round(time.time()) - round(self.start_time)

        # Record Position
        # --------------------------------------------------
        self.path['seconds'].append(seconds)
        self.path['north'  ].append( self.local_position[0])
        self.path['east'   ].append( self.local_position[1])
        self.path['down'   ].append(-self.local_position[2])

        # Record Velocity
        # --------------------------------------------------
        self.velocity['seconds'].append(seconds)
        self.velocity['north'  ].append(self.local_velocity[0])
        self.velocity['east'   ].append(self.local_velocity[1])
        self.velocity['down'   ].append(self.local_velocity[2])


    def calculate_box(self):
        """
        """

        print("[INFO] calculating flight path")

        self.all_waypoints.append((self.distance,             0, 3, 0))
        self.all_waypoints.append((self.distance, self.distance, 3, 0))
        self.all_waypoints.append((            0, self.distance, 3, 0))
        self.all_waypoints.append((            0,             0, 3, 0))


    def arming_transition(self):
        """
        """

        print("[INFO] arming transition")

        self.take_control()
        self.arm()

        self.set_home_position(
            self.global_position[0],
            self.global_position[1],
            self.global_position[2]
        )

        self.flight_state = States.ARMING


    def takeoff_transition(self):
        """
        """

        print(f"[INFO] takeoff transition")

        target_altitude         = 3.0
        self.target_position[2] = target_altitude

        self.takeoff(target_altitude)

        self.flight_state = States.TAKEOFF


    def waypoint_transition(self):
        """
        """

        print("[INFO] waypoint transition")
        self.target_position = self.all_waypoints.pop(0)

        print(f"[INFO] target position {self.target_position}")
        self.cmd_position(*self.target_position)

        self.flight_state = States.WAYPOINT


    def landing_transition(self):
        """
        """

        print("[INFO] landing transition")
        self.land()

        print(
            f"[INFO] distance from start:\n"
            f"       meters north: {round(self.start_position[0] - self.local_position[0], 2)}\n"
            f"       meters east:  {round(self.start_position[1] - self.local_position[1], 2)}"
        )

        self.flight_state = States.LANDING


    def disarming_transition(self):
        """
        """

        print("[INFO] disarm transition")

        self.disarm()
        self.release_control()

        self.flight_state = States.DISARMING


    def gen_plots(self):
        print("[INFO] generating path plot")

        self.v.line(
            X    = np.array(self.path['east']),
            Y    = np.array(self.path['north']),
            win  = 'path',
            opts = dict(
                title  = f"Mission Path (Start at [0, 0])",
                xlabel = 'East',
                ylabel = 'North'
            )
        )

        print("[INFO] generating altitude plot")

        self.v.line(
            X    = np.array(self.path['seconds']),
            Y    = np.array(self.path['down']),
            win  = 'altitude',
            opts = dict(
                title  = f"Mission Altitude",
                xlabel = 'Time Step',
                ylabel = 'Meters'
            )
        )

        for stat in list(self.velocity.keys())[1:]:
            print(f"[INFO] generating velocity {stat.capitalize()} plot")
            self.v.line(
                X    = np.array(self.velocity['seconds']),
                Y    = np.array(self.velocity[stat]),
                win  = stat,
                opts = dict(
                    title  = f"Mission Velocity ({stat.capitalize()})",
                    xlabel = 'Time Step',
                    ylabel = 'Meters / Second'
                )
            )


    def manual_transition(self):
        """
        """

        print("[INFO] manual transition")

        self.release_control()
        self.stop()

        self.in_mission   = False
        self.flight_state = States.MANUAL


    def start(self):
        """
        """

        print("[Info] ceating log file")
        self.start_log("logs", "navLog.txt")

        print("[INFO] starting connection")
        self.connection.start()

        if self.use_visdom == 'true':
            print("[INFO] generating plots")
            self.gen_plots()

        print("[Info] closing log file")
        self.stop_log()

        print(f"[INFO] ellapsed time: {round(time.time() - self.start_time, 2)} seconds")



# Application 
# ==================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--port',       type = int, default = 5760,        help = 'port number [default = 5760]')
    parser.add_argument('--host',       type = str, default = '127.0.0.1', help = "host address [default = '127.0.0.1']")
    parser.add_argument('--use_visdom', type = str, default = 'false',     help = "use visdom by setting to 'true' if server running [default = 'false']")
    parser.add_argument('--distance',   type = int, default = 10,          help = "meters to travel along each edge of the square path [default = 10]")

    args  = parser.parse_args()
    conn  = MavlinkConnection(f"tcp:{args.host}:{args.port}", threaded = False, PX4 = False)
    drone = BackyardFlyer(
        conn,
        args.use_visdom,
        args.distance
    )

    time.sleep(2)

    drone.start()