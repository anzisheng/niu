#frog: head
#arm: elbow_right
#run
#
global score_all
score_all = 0


global count_down
count_down = False


global record_time
record_time = 5

fps_time = 0
f = 0
global startTime
startTime = 0.0



global detect_model
detect_model = None
global tracker
tracker = None
global pose_model
pose_model = None
global action_model
action_model = None

global GROUND
GROUND = 0
global GROUND_DISTANCE_SMALL
GROUND_DISTANCE_SMALL = 4

global GROUND_DISTANCE_MIDDLE
GROUND_DISTANCE_MIDDLE = 8

global GROUND_DISTANCE_LARGE
GROUND_DISTANCE_LARGE = 12

global LFoot_Up
LFoot_Up = False
global RFoot_Up
RFoot_Up = False

global HEAD_ABOVE
HEAD_ABOVE = 6
global HEAD_OFFSET_X
HEAD_OFFSET_X = 6

global NUM_Peron
NUM_Peron = 0

global MAX_Peron
MAX_Peron = 5

### frog action
#head: 0
Head_Part = 0
global Head_x
Head_x = []
global Head_y
Head_y = []

RShoulder_Part = 1
global RShoulder_y
RShoulder_y = []

#elbow_right : 3
global elbow_right_x
elbow_right_x = []
global elbow_right_y
elbow_right_y = []


#右手
RWist_Part = 5
global RWrist_y
RWrist_y = []

#right hip : 7
global hip_right_x
hip_right_x = []
global hip_right_y
hip_right_y = []


global LKnee_x
LKnee_x = []
global LKnee_y
LKnee_y = []

#右膝盖
RKnee_Part = 9
global RKnee_x
RKnee_x = []
global RKnee_y
RKnee_y = []

#right angle 11
global RAnkle_x
RAnkle_x = []
global RAnkle_y
RAnkle_y = []

#left angle 12
global LAnkle_x
LAnkle_x = []
global LAnkle_y
LAnkle_y = []

# global LAnkle_x
# LAnkle_x = []
# global LAnkle_y
# LAnkle_y = []
# global RAnkle_x
# RAnkle_x = []
# global RAnkle_y
# RAnkle_y = []



global FIRST
FIRST = []
global FirstFrame
FirstFrame = True
global action_times
action_times = []
global action_Lock
action_Lock = []

global Lock_L_Foot
Lock_L_Foot = []

global Lock_R_Foot
Lock_R_Foot = []


