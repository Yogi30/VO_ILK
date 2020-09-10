import numpy as np
import cv2

class CameraModel:
    def __init__(self,params):
        self.width = params['width']
        self.height = params['height']
        self.focal_x = params['focal_x']
        self.focal_y = params['focal_y']
        self.cx = params['cx']
        self.cy = params['cy']
        self.distort = (abs(params['k1']) > 0.0000001)
        self.distort_params = [params['k1'], params['k2'], params['p1'], params['p2'], params['k3']]

class VO:
    def __init__(self, cam, poses):
        self.curr_stage = 0
        self.cam = cam

        self.focal_length = cam.focal_x
        self.cam_center = (cam.cx, cam.cy)

        self.frame_latest = None
        self.frame_prev = None
        self.rot = None
        self.trans = None

        self.detector = cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression=True)
        self.keypoints_curr = None
        self.keypoints_prev = None

        self.truePos_x = 0
        self.truePos_y = 0
        self.truePos_z = 0

        self.frame_stage = 1
        self.default_frame = 0
        self.first_frame = 1
        self.second_frame = 2

        self.lk_params = dict(winSize  = (21, 21), 
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        self.kMinNumFeature = 1500
        with open(poses) as f:
            self.poses = f.readlines()
    
    def update_stage(self, img, img_no):
        if not (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width):
            print("input error")
        self.frame_latest = img
        if self.frame_stage == self.default_frame:
            self.processDefFrame(img_no)
        elif self.frame_stage == self.first_frame:
            self.processFirstFrame()
        elif self.frame_stage == self.second_frame:
            self.processSecondFrame()
        self.frame_prev = self.frame_latest


    def processDefFrame(self, imgNo):
        self.keypoints_prev, self.keypoints_curr = self.feature_tracking(self.frame_prev, self.frame_latest, self.keypoints_prev)
        E_Mat, mask = cv2.findEssentialMat(self.keypoints_curr, self.keypoints_prev, focal = self.cam.focal_x, pp = (self.cam.cx, self.cam.cy), 
                                            method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
        n, rot, trans, mask = cv2.recoverPose(E_Mat, self.keypoints_curr, self.keypoints_prev, focal = self.cam.focal_x, pp = (self.cam.cx, self.cam.cy))
        scale = self.getTruePose(imgNo)
        if scale > 0.1:
            self.trans = self.trans + scale * self.rot.dot(trans) 
            self.rot = rot.dot(self.rot)
        
        if self.keypoints_prev.shape[0] < self.kMinNumFeature:
            self.keypoints_curr = self.detector.detect(self.frame_latest)
            self.keypoints_curr = np.array([i.pt for i in self.keypoints_curr], dtype=np.float32)
        self.keypoints_prev = self.keypoints_curr


    def processFirstFrame(self):
        self.keypoints_prev = self.detector.detect(self.frame_latest)
        # keypoint detectors inherit the FeatureDetector interface
        self.keypoints_prev = np.array([i.pt for i in self.keypoints_prev], dtype=np.float32)
        self.frame_stage = self.second_frame

    def processSecondFrame(self):
        # import pdb;pdb.set_trace()
        self.keypoints_prev, self.keypoints_curr = self.feature_tracking(self.frame_prev, self.frame_latest, self.keypoints_prev)
        E_Mat, mask = cv2.findEssentialMat(self.keypoints_curr, self.keypoints_prev, focal = self.cam.focal_x, pp = (self.cam.cx, self.cam.cy), 
                                            method = cv2.RANSAC, prob = 0.999, threshold = 1.0)
        n, self.rot,self.trans, mask = cv2.recoverPose(E_Mat, self.keypoints_curr, self.keypoints_prev, focal = self.cam.focal_x, pp = (self.cam.cx, self.cam.cy))
        self.frame_stage = self.default_frame
        self.keypoints_prev = self.keypoints_curr

    def feature_tracking(self, img_prev, img_curr, keypoints_prev):
        # An example using the Lucas-Kanade optical flow algorithm can be 
        # found at opencv_source_code/samples/python2/lk_track.py
        kp2, status, err = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, keypoints_prev, None, **self.lk_params)
        # status – output status vector; each element of the vector is set to 1 if the flow for 
        # the corresponding features has been found, otherwise, it is set to 0.
        # kp2 - output vector of 2D points (with single-precision floating-point coordinates)
        # containing the calculated new positions of input features in the second image
        status = status.reshape(status.shape[0])
        # keeping only keypoints found in current and prev frames
        kp1 = keypoints_prev[status == 1]
        kp2 = kp2[status == 1]
        
        return kp1, kp2
    
    def getTruePose(self, frame_id):
        ss = self.poses[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.poses[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.truePos_x, self.truePos_y, self.truePos_z = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

if __name__ == "__main__":
    params = {}
    # Camera parameters - KITTI Dataset
    params['width'],params['height'] = 1241.0, 376.0
    params['focal_x'], params['focal_y'] = 718.8560, 718.8560
    params['cx'], params['cy'] = 607.1928, 185.2157
    params['k1'], params['k2'], params['p1'], params['p2'], params['k3'] = 0.0,0.0,0.0,0.0,0.0
    
    cam = CameraModel(params)
    vo = VO(cam, '/home/yogesh/monoVO-python/data_odometry_poses/dataset/poses/00.txt')
    traj = np.zeros((600,600,3), dtype=np.uint8)

    for img_no in range(4541):
        img = cv2.imread('/home/yogesh/monoVO-python/data_odometry_gray/dataset/sequences/00/image_0/'+str(img_no).zfill(6)+'.png', 0)
        
        vo.update_stage(img, img_no)
        trans = vo.trans

        if img_no > 2 :
            x,y,z = trans[0], trans[1], trans[2]
        else:
            x,y,z = 0.0, 0.0, 0.0
        
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(vo.truePos_x) + 290, int(vo.truePos_z) + 90
        cv2.circle(traj, (draw_x,draw_y), 1, (img_no*255/4540,255-img_no*255/4540,0), 1)
        cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        # text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        # cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.imshow('Feed From Camera', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)
    cv2.imwrite('map.png', traj)

        

