import re
import cv2
import time
import math
import torch
import numpy as np
import  globalvar
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255,255,255)

#global head_x
#global head_y
global FIRST
#global frog_times
#global Lock
#Lock = False
FIRST = True
head_x = 0
head_y = 0
#frog_times = 0
##########################
# #NUM_PERSON = 5
#exercise
global elbow_x
global elbow_y
global Exer_FIRST
#elbow_x = [] #* NUM_PERSON
#elbow_x = [] #* NUM_PERSON
Exer_FIRST = True
##########################

"""COCO_PAIR = [(0, 1), (0, 2), (1, 3), (2, 4),  # Head
             (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (17, 11), (17, 12),  # Body
             (11, 13), (12, 14), (13, 15), (14, 16)]"""
COCO_PAIR = [(0, 13), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (13, 7), (13, 8),  # Body
             (7, 9), (8, 10), (9, 11), (10, 12)]
#COCO_PAIR = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]

POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

MPII_PAIR = [(8, 9), (11, 12), (11, 10), (2, 1), (1, 0), (13, 14), (14, 15), (3, 4), (4, 5),
             (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)]

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def collate_fn_list(batch):
    img, inp, im_name = zip(*batch)
    img = collate_fn(img)
    im_name = collate_fn(im_name)

    return img, inp, im_name


def draw_single(frame, pts, joint_format='coco'):
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]

    else:
        NotImplementedError

    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
    for n in range(pts.shape[0]):
        #n = pts.shape[0] -1 #13
        # if n not in [0, 1, 2, 13 ]:
        #     continue

        if pts[n, 2] <= 0.05:
            continue
        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        #n = 0
        cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)
        #cv2.circle(frame, (cor_x, cor_y), 3,(255,255,255), 4)
        #cv2.putText(frame, str(n), (cor_x, cor_y))
        frame = cv2.putText(frame, str(n), (cor_x, cor_y), cv2.FONT_HERSHEY_DUPLEX,
                             0.3, (255, 0, 0), 1)

    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(frame, start_xy, end_xy, line_color[i], int(1*(pts[start_p, 2] + pts[end_p, 2]) + 1))
            #cv2.line(frame, start_xy, end_xy, (255,255,255), int(1 * (pts[start_p, 2] + pts[end_p, 2]) + 1))
    return frame


def handle_run(pts, index):
    if (globalvar.FIRST[index]):
        globalvar.FIRST[index] = False
        globalvar.LAnkle_x[index], globalvar.LAnkle_y[index] = pts[12, 0:2]
        globalvar.RAnkle_x[index], globalvar.RAnkle_y[index] = pts[11, 0:2]
        # globalvar.GROUND = max(globalvar.RAnkle_y[index], globalvar.LAnkle_y[index])

        # not the first frame, do normal logic
    globalvar.GROUND = max(pts[11, 1], pts[12, 1])
    if (pts[12, 1] < globalvar.GROUND - globalvar.GROUND_DISTANCE_SMALL):
        if not globalvar.Lock_L_Foot[index]:
            if (not globalvar.RFoot_Up):
                globalvar.Lock_L_Foot[index] = True
                globalvar.LFoot_Up = True
                # globalvar.action_times[index] += 1
        else:
            globalvar.Lock_L_Foot[index] = False
    if (pts[11, 1] < globalvar.GROUND - globalvar.GROUND_DISTANCE_SMALL):
        if not globalvar.Lock_R_Foot[index]:
            if (not globalvar.LFoot_Up):
                globalvar.Lock_R_Foot[index] = True
                globalvar.RFoot_Up = True
                globalvar.action_times[index] += 1
        else:
            globalvar.Lock_R_Foot[index] = False
def hanle_jump_foot(pts, index):
    pass

#############################################################
# 抬起动作
# 规则：右手（5）低于膝关节（9）然后高于为一次
# test video: /home/anzisheng/projectx/xiaoniu_test/data_test/lift/小32-3 抬起担架（下蹲）.mp4
# --camera "/home/anzisheng/projectx/xiaoniu_test/data_test/lift/小32-3 抬起担架（下蹲）.mp4" --save_out out_video/record.mp4
def handle_lift(pts, index):
    if (globalvar.FIRST[index]):
        globalvar.FIRST[index] = False
        globalvar.RKnee_y[index] = pts[globalvar.RKnee_Part, 1]

        if pts[globalvar.RWist_Part,1] < globalvar.RKnee_y[index]:
            if not globalvar.action_Lock[index]:
                globalvar.action_Lock[index] = True
                #globalvar.frog_times[index] += 1
                globalvar.action_times[index] += 1
                #print("times:{}".format(globalvar.action_times[index]))
        else:
            globalvar.action_Lock[index] = False

############################################################
#规则：右手（5）高于右肩膀(1)
#"/home/anzisheng/projectx/xiaoniu_test/data_test/arm/小32-2 指挥交通（挥动手臂）.mp4"
# --camera "/home/anzisheng/projectx/xiaoniu_test/data_test/arm/小32-2 指挥交通（挥动手臂）.mp4" --save_out out_video/arm.mp4
def handle_arm(pts, index):
    # right hand: 5
    if (globalvar.FIRST[index]):
        globalvar.FIRST[index] = False
        globalvar.RShoulder_y[index] = int(pts[globalvar.RShoulder_Part, 1])
        #globalvar.head_y[index] = int(pts[0, 1])
        # globalvar.hand_right_x[index], \
        #globalvar.RWrist_y[index] = pts[globalvar.RWist_Part, 1]
        #globalvar.hip_right_x[index], globalvar.hip_right_y[index] = pts[7, 0:2]

    # start_0 = (globalvar.head_x[index] - globalvar.HEAD_OFFSET_X, globalvar.head_y[index] - globalvar.HEAD_ABOVE)
    # start_1 = (globalvar.head_x[index] + globalvar.HEAD_OFFSET_X, globalvar.head_y[index] - globalvar.HEAD_ABOVE)
    # cv2.line(frame, start_0, start_1, (0, 0, 0), 1)

    if pts[globalvar.RWist_Part, 1] < globalvar.RShoulder_y[index]:
        if not globalvar.action_Lock[index]:
            globalvar.action_Lock[index] = True
            globalvar.action_times[index] += 1
            #print("times:{}".format(globalvar.action_times[index]))
    elif (pts[5, 1] > globalvar.hip_right_y[index]):
        globalvar.action_Lock[index] = False


def draw_single_action(frame, pts, action_name,index,joint_format='coco'):
    #return frame
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]
    else:
        NotImplementedError
    """
    if(action_name == "exercise"):
        if FIRST:
            head_x = int(pts[0, 0])
            head_y = int(pts[0, 1])
            FIRST = False
    """


############################################
    #run:
    #LAngle = 12
    #RAngle = 11
    if (action_name == "run"):
        handle_run(pts, index)
    if(action_name == "lift"):
        handle_lift(pts, index)
    ######################################################
    #frog:
    if(action_name == "frog"):
        print("index: {}".format(index))
        print("globalvar.FIRST: ")
        print(globalvar.FIRST)

        if (globalvar.FIRST[index]):
            globalvar.Head_x[index] = int(pts[0, 0])
            globalvar.Head_y[index] = int(pts[0, 1])
            globalvar.FIRST[index] = False

        start_0 = (globalvar.Head_x[index] - globalvar.HEAD_OFFSET_X, globalvar.Head_y[index] - globalvar.HEAD_ABOVE)
        start_1 = (globalvar.Head_x[index] + globalvar.HEAD_OFFSET_X, globalvar.Head_y[index] - globalvar.HEAD_ABOVE)
        #fixed line
        #cv2.line(frame, start_0, start_1, (0, 0, 0), 1)

        if pts[0,1] < globalvar.Head_y[index] - globalvar.HEAD_ABOVE :
            if not globalvar.action_Lock[index]:
                globalvar.action_Lock[index] = True
                #globalvar.frog_times[index] += 1
                globalvar.action_times[index] += 1
                #print("times:{}".format(globalvar.action_times[index]))
        else:
            globalvar.action_Lock[index] = False
        print("times:{}".format(globalvar.action_times[index]))
#--------------------------------------------------------
    #right hand: 5
    if(action_name == "arm"):
        #handle_arm(pts,index)
        pass
        #
        # if (globalvar.FIRST[index]):
        #     globalvar.FIRST[index] = False
        #     globalvar.head_x[index] = int(pts[0, 0])
        #     globalvar.head_y[index] = int(pts[0, 1])
        #     #globalvar.hand_right_x[index], \
        #     globalvar.RWrist_y[index] = pts[5, 1]
        #     globalvar.hip_right_x[index], globalvar.hip_right_y[index] = pts[7, 0:2]
        #
        #
        # start_0 = (globalvar.head_x[index] - globalvar.HEAD_OFFSET_X, globalvar.head_y[index] - globalvar.HEAD_ABOVE)
        # start_1 = (globalvar.head_x[index] + globalvar.HEAD_OFFSET_X, globalvar.head_y[index] - globalvar.HEAD_ABOVE)
        # cv2.line(frame, start_0, start_1, (0, 0, 0), 1)
        #
        #
        # if  pts[5,1] < globalvar.head_y[index] :
        #     if not globalvar.action_Lock[index]:
        #         globalvar.action_Lock[index] = True
        #         globalvar.action_times[index] += 1
        #         print("times:{}".format(globalvar.action_times[index]))
        # elif (pts[5,1] >  globalvar.hip_right_y[index]) :
        #     globalvar.action_Lock[index] = False

########################################
    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[5, :] + pts[6, :]) / 2, 0)), axis=0) #add neck, slow down slightly
    #pts[:, 0] = pts[:,0] / frame.shape[0]
    #pts[:, 1] = pts[:,1] / frame.shape[1]
    #centering
    #pts[0:2] = pts[0:2] - 0.5


    for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.05:
            continue
        cor_x, cor_y = (pts[n, 0]), (pts[n, 1])
        part_line[n] = (cor_x, cor_y)

        #cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)
        #cv2.circle(frame, (cor_x, cor_y), 3,(255,255,255), 1)

        #cv2.putText(frame, str(n), (cor_x, cor_y))
        #frame = cv2.putText(frame, str(n), (cor_x, cor_y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 0, 0), 1)

    H, W, c = frame.shape
    scale_factor = 2 * H / 1080
    #scale_factor = 1

    self_link = [(i, i) for i in range(18)]
    neighbor_link = [
        (10, 8),    #(4, 3) RWrist， RElbow
        (8, 6),     #(3, 2) RElbow， RShoulder
        (9, 7),     #(7, 6),LWrist  LElbow
        (7, 5),     #(6, 5) LElbow, LShoulder
        (15, 13),   #(13, 12), LAnkle, LKnee
        (13, 11),   #(12, 11), LKnee,  LHip
        (16, 14),   #(10, 9), RAnkle, RKnee
        (14, 12),   #(9, 8), RKnee, RHip
        (11, 5),    #(11, 5),LHip, LShoulder
        (12, 6),    #(8, 2),RHip, RShoulder
        (5, 17),    #(5, 1),LShoulder, Neck
        (6, 17),    #(2, 1),RShoulder, Neck
        (0, 17),    #(0, 1),Nose, Neck
        (1, 0),     #(15, 0),LEye Nose
        (2, 0),    #(14, 0), REye, Nose
        (3, 1),     #(17, 15),LEar，LEye
        (4, 2)    #(16, 14) REar，REye
    ]
    l_pair = self_link + neighbor_link

    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]

            #for i, j in edge:
            xi = start_xy[0]#pose[0, t, i, m]
            yi = start_xy[1]#pose[1, t, i, m]
            xj = end_xy[0]#pose[0, t, j, m]
            yj = end_xy[1]#pose[1, t, j, m]
            if xi + yi == 0 or xj + yj == 0:
                    continue
            else:
                xi = int((xi + 0.5))# * W)
                yi = int((yi + 0.5))# * H)
                xj = int((xj + 0.5))# * W)
                yj = int((yj + 0.5)) #* H)
            cv2.line(frame, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil( 6 * scale_factor *0.3)))


            #cv2.line(frame, start_xy, end_xy, line_color[i], int(1*(pts[start_p, 2] + pts[end_p, 2]) + 1))
            #cv2.line(frame, start_xy, end_xy, (255,255,255), int(1 * (pts[start_p, 2] + pts[end_p, 2]) + 1))
    return frame

def draw_single_color(frame, pts, joint_format='coco'):
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]

    else:
        NotImplementedError

    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
    for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.05:
            continue
        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)
        #cv2.circle(frame, (cor_x, cor_y), 3,(255,255,255), -1)

    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(frame, start_xy, end_xy, line_color[i], int(1*(pts[start_p, 2] + pts[end_p, 2]) + 1))
            #cv2.line(frame, start_xy, end_xy, (255,255,255), int(1 * (pts[start_p, 2] + pts[end_p, 2]) + 1))
    return frame

def vis_frame_fast(frame, im_res, joint_format='coco'):
    """
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    """
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED,BLUE,BLUE]
    else:
        NotImplementedError

    #im_name = im_res['imgname'].split('/')[-1]
    img = frame
    for human in im_res:  # ['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[1, :]+kp_preds[2, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[1, :]+kp_scores[2, :]) / 2, 0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2*(int(kp_scores[start_p] + kp_scores[end_p])) + 1)
    return img

def vis_frame(frame, im_res, joint_format='coco'):
    """
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    """
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    im_name = im_res['imgname'].split('/')[-1]
    img = frame
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width/2), int(height/2)))
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :]+kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :]+kp_scores[6, :]) / 2, 0)))
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x/2), int(cor_y/2))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x/2), int(cor_y/2)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX),int(mY)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                #cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(0, min(1, 0.5*(kp_scores[start_p] + kp_scores[end_p])))
                img = cv2.addWeighted(bg, transparency, img, 1-transparency, 0)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img


def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval
