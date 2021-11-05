# Argument Constants
# ==================================================
CONN_PORT     = 5760       
CONN_HOST     = "127.0.0.1"
DEFAULT_NORTH = 10
DEFAULT_EAST  = 10
USE_VISDOM    = "false"



# Drone Constants
# ==================================================
CONN_TIMEOUT = 60

TARGET_ALTITUDE = 20
SAFETY_DISTANCE = 5



# MAP Constants
# ==================================================
METERS_2_LAT = 9.00900900900901e-06
METERS_2_LON = METERS_2_LAT * 1.2

DEADBAND_RADIUS   = 3
DEADBAND_ALTITUDE = 0.8



# I/O Constants
# ==================================================
CITY_MAP  = "colliders.csv"
CSV_DELIM = ","

LOG_DIR = "logs"
LOG_FILE = "nav_log.txt"


# Visdom Constants
# ==================================================
VISDOM_LINK = "http://localhost:8097"