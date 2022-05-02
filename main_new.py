import os
from datetime import datetime
import json
import requests

import globalvar
import cv2
import time
import torch
import argparse
import numpy as np


from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single, draw_single_action

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from PIL import Image, ImageDraw, ImageFont
#--camera data_test/frog.mp4
#source = './Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect

#source = '../Data/falldata/Home/Videos/video (1).avi'

source = '0'
#--camera 0
#--camera data_test/exercise.avi

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,

                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def UpFile(Url, FilePath, data):
    '''
    用于POST上传文件以及提交参数
    @ Url 上传接口
    @ FilePath 文件路径
    @ data 提交参数 {'key':'value', 'key2':'value2'}
    '''
    files = {'file': open(FilePath, 'rb')}
    result = requests.post(Url, files=files, data=data)
    return result


def upload_result(filename, gameId, schoolId, gradeId, score):
    headers = {'content-type': 'application/json'}
    dataURL = 'https://wisdom.prefootball.cn/wisdom_school/ai_game_log/add/scoreinfo' #https://wisdom.prefootball.cn/wisdom_school/
    #         https: // wisdom.prefootball.cn / wisdom_school /
    requestData = {
        "wisdomAigameId": gameId,
        "schoolId": schoolId,
        "gradeId": gradeId,
        "score": score,
        "challengeNumber": "3"
    }
    ret = requests.post(dataURL, json=requestData, headers=headers)
    if ret.status_code == 200:
        print(ret.text)
        s1 = json.loads(ret.text)
        id = s1["data"]["id"]
        videoURL = 'https://wisdom.prefootball.cn/wisdom_school/ai_game_log/add/videoinfo'

        # 上传接口
        url = videoURL
        # 需提交的参数
        data = {'id': id}
        # 需上传的文件路径
        file = filename #self.dest_filename#  args_global.save_out#'record.mp4'
        r = UpFile(url, file, data)
        # 打印返回的值
        print(r.text)
        return

        # files = {'file': open('record.mp4','rb')}  # 此处是重点！我们操作文件上传的时候，把目标文件以open打开，然后存储到变量file里面存到一个字典里面
        # f = open('record.mp4','rb')
        # upload_data = {"id": id, "file": f}
        # headers = {'content-type': 'multipart/form-data'}
        # ret = requests.post(videoURL, files=files, data=upload_data,
        #                            headers=headers)  ##此处是重点！我们操作文件上传的时候，接口请求参数直接存到upload_data变量里面，在请求的时候，直接作为数据传递过去
        # post_file_url = videoURL
        # headers = {
        #     }
        # files = {"files": open('record.mp4', 'rb'), "Content-Type": "multipart/form-data",
        #          "Content-Disposition": "form-data", "filename": 'record.mp4', "id": id}
        # post_response = requests.post(url=post_file_url, headers=headers, files=files, verify=False)
        # print('post_response:{}'.format(post_response))


        # requestData = {
        #     "id": id,
        #     "file": open('D:\\test_data\\summer_test_data_05.txt','rb')}#,
        # }
        # ret = requests.post(dataURL, json=requestData, headers=headers)
        if ret.status_code == 200:
            #print(ret.text)
            return True




#--camera data_test/exercise.avi
#--camera data_test/exercise.avi
if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,   #required=True,  #default='0', source
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=1024,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--gameId', type=str, default='1',
                     help='gameId.')
    par.add_argument('--schoolId', type=str, default='1',
                     help='school Id.')
    par.add_argument('--gradeId', type=str, default='1',
                     help='grade Id.')

    par.add_argument('--device', type=str, default='cuda',#cuda
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    gameId = int(args.gameId)
    print(gameId)
    gradeId = int(args.gradeId)
    print(gradeId)
    schoolId = int(args.schoolId)
    print(schoolId)

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=10)

    # Actions Estimate.
    action_model = TSSTG(device=args.device)

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    outvid = True
    #if args.save_out != '':
    outvid = True
    now = datetime.now()  # current date and time
    # save self.dest_image
    dest_filename = now.strftime("%H_%M_%S") + '.mp4'
    codec = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    writer = cv2.VideoWriter()
    writer.open(dest_filename, codec, 30, (640,480), True) #(inp_dets * 4, inp_dets * 2)
    print(args.save_out)

    fps_time = 0
    f = 0
    record_time = 60
    #First_all = True
    startTime = datetime.now()

    count_down = True#False
    while record_time > 0:
        if not cam.grabbed():
            break

        f += 1
        frame = cam.getitem()
        image = frame.copy()

        #skeleton = frame*0
        #skeleton = cv2.imread("background/pet_store.png", cv2.IMREAD_UNCHANGED)
        #skeleton = cv2.resize(skeleton, (frame.shape[1], frame.shape[0]))

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)


        # detected_fix = []
        # detected_fix.append([204, 460, 339,740, 8.7664e-01, 1.0000e+00,1])
        # detected_fix.append([438,458, 580, 790,  8.5063e-01, 1.0000e+00,1])
        # detected_fix.append([650, 460, 758, 736,7.3046e-01, 1.0000e+00,1])

        #detected_fix.from_numpy(detected_fix)
        #fix2 = np.array(detected_fix)
        #detected= torch.from_numpy(fix2)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # if any(globalvar.FIRST):
        #     globalvar.FIRST = [True] * globalvar.NUM_Peron


        #globalvar.FIRST.All

        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            #this code is the most appensive
            #This code is key anzs
            #anzisheng
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
            #
            # # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.

            if args.show_detected:
                #for bb in detected_fix[:]: #, 0:5]:
                for bb in detected[: , 0:5]:
                    frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 1)
                    #frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 4)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)
        globalvar.NUM_Peron = len(tracker.tracks)
        if (globalvar.NUM_Peron > 0):
            if (globalvar.FirstFrame == True):
                globalvar.FirstFrame = False
                #globalvar.NUM_Peron = len(tracker.tracks)

                globalvar.Head_x = [0] * globalvar.MAX_Peron #0
                globalvar.Head_y = [0] * globalvar.MAX_Peron

                globalvar.RShoulder_y = [0] * globalvar.MAX_Peron #1

                globalvar.elbow_right_x = [0] * globalvar.MAX_Peron
                globalvar.elbow_right_y = [0] * globalvar.MAX_Peron

                globalvar.RWrist_y = [0] * globalvar.MAX_Peron
                #globalvar.hand_right_x = [0] * globalvar.NUM_Peron

                #9
                globalvar.RKnee_x = [0] * globalvar.MAX_Peron
                globalvar.RKnee_y = [0] * globalvar.MAX_Peron
                #10
                globalvar.LKnee_x = [0] * globalvar.MAX_Peron
                globalvar.LKnee_y = [0] * globalvar.MAX_Peron

                globalvar.hip_right_x = [0] * globalvar.MAX_Peron
                globalvar.hip_right_y = [0] * globalvar.MAX_Peron

                globalvar.action_times = [0] * globalvar.MAX_Peron
                globalvar.FIRST = [True] * globalvar.MAX_Peron

                globalvar.action_Lock = [False] * globalvar.MAX_Peron
                globalvar.Lock_L_Foot = [False] * globalvar.MAX_Peron
                globalvar.Lock_R_Foot = [False] * globalvar.MAX_Peron


                globalvar.LAnkle_x = [0] * globalvar.MAX_Peron
                globalvar.LAnkle_y = [0] * globalvar.MAX_Peron
                globalvar.RAnkle_x = [0] * globalvar.MAX_Peron
                globalvar.RAnkle_y = [0] * globalvar.MAX_Peron

            # Predict Actions of each track.

        for i, track in enumerate(tracker.tracks):
            if( i >= globalvar.MAX_Peron) :
                continue

            # if(globalvar.FIRST[i] == False):
            #     globalvar.FIRST[i] = False
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                #out = action_model.predict(pts, frame.shape[:2])
                #action_name = action_model.class_names[out[0].argmax()]
                action = 'pending..'#'{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            #     if action_name == 'Fall Down':
                #     clr = (255, 0, 0)
                # elif action_name == 'Lying Down':
                #     clr = (255, 200, 0)
            target_action = "frog"
            #target_action = "arm"
            #target_action = "lift"

            #target_action = "exercise"
            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    #frame = draw_single(frame, track.keypoints_list[-1])
                    #skeleton = draw_single(skeleton, track.keypoints_list[-1])
                    #if target_action == "frog":
                    frame = draw_single_action(frame, track.keypoints_list[-1], target_action, i) #skeleton
#                        First_frame[i] = False
#                     if target_action == "frog":
#                         skeleton = draw_single_action(skeleton, track.keypoints_list[-1], target_action)


                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                #keleton = cv2.rectangle(skeleton, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

                #frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2)
                #frame = cv2.putText(skeleton, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,0.4, clr, 1)

                #frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 2)
                #keleton = cv2.putText(skeleton, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.3, clr, 1)
                frame = cv2.putText(frame, str(globalvar.action_times[i]), (bbox[0] + 5, bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
                #frame = cv2.putText(frame, str(record_time), (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0),  1)

                if (count_down):
                    if record_time > 0:  # < totalSec:
                        # draw the Nth second on each frame
                        # till one second passes
                        # cv2.putText(img=frame,
                        #             text=str(record_time),  # strSec[nSecond],
                        #             org=(int(240 / 2 - 20), int(480 / 2)),
                        #             fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        #             fontScale=6,
                        #             color=(255, 0, 0),
                        #             thickness=5
                        #             )
                        timeElapsed = (datetime.now() - startTime).total_seconds()
                        if timeElapsed >= 1:
                            record_time -= 1
                            #                print 'nthSec:{}'.format(nSecond)
                            timeElapsed = 0
                            startTime = datetime.now()

                        #    record_time -= 1
                else:
                    count_down = False
                    record_time = 60

        # Show Frame.
        #frame = cv2.putText(frame, "Time: "+ str(record_time), (20, 85), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 1)
        frame = cv2ImgAddText(frame, "时间: "+ str(record_time) + " 秒", 20, 85)
        globalvar.score_all = sum(globalvar.action_times) * 2 + globalvar.NUM_Peron * 5
        if(globalvar.score_all > 100):
            globalvar.score_all = 100

        #frame = cv2.putText(frame, "Score: " + str(globalvar.score_all), (810, 85), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 1)
        frame = cv2ImgAddText(frame, "得分: " + str(globalvar.score_all), 810, 85)#, cv2.FONT_HERSHEY_COMPLEX, 1.2, #(255, 0, 0), 1)
        #frame = cv2.resize(frame, (0, 0), fx=1.8, fy=1.05)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #cv2.imwrite("debug.jpg",frame)
        frame = frame[:, :, ::-1]


        # skeleton port
        #skeleton = frame * 0
        #draw skeleton
        # skeleton = cv2.resize(skeleton, (0, 0), fx=2., fy=2.)
        # skeleton = skeleton[:,:, ::-1]
        fps_time = time.time()
        #frame = np.concatenate((frame, skeleton), axis=1)
        cv2.imshow('frame', frame)

        if outvid:
            frame = cv2.resize(frame, (640, 480))
            writer.write(frame)
            #print("write ....")

        if cv2.waitKey(1) & 0xFF == ord('s'):
            count_down = True
            startTime = datetime.now()
        #skeleton = cv2.putText(skeleton, str(show_time), (20, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
        upload_result(dest_filename, gameId, schoolId, gradeId, globalvar.score_all)
        print("writer over")
    cv2.destroyAllWindows()
