#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Int32MultiArray, Header, ColorRGBA
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
import cv2
from BirdEyeView import BirdEyeView
from LaneDetector import LaneDetector
import time

WEIGHT = 800
def bilateralfilter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def image_processing(bev, img):
    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 10)
#    cv2.imshow("1", blur)

    #blur = bilateralfilter(gray)
    canny = cv2.Canny(blur, 80, 90)
#    cv2.imshow("2", canny)

    #Delete Here
    #ret, Binary_img = cv2.threshold(blur,140,255,cv2.THRESH_BINARY)

    warped_frame = bev.warpPerspect(canny)
#    warped_frame = bev.warpPerspect(Binary_img)
#    cv2.imshow("3", warped_frame)

    #warped_frame = bev.warpPerspect(blur)
    ret, Binary_img = cv2.threshold(warped_frame,140,255,cv2.THRESH_BINARY)
    cv2.rectangle(Binary_img, (img_w / 6, 2 * img_h / 3), (5 * img_w / 6,3 * img_h / 4), (255, 255, 0), 3)
    cv2.imshow("4", Binary_img)
    return Binary_img

def show_text_in_rviz(marker_publisher, text):
    marker = Marker(
    type=Marker.TEXT_VIEW_FACING,
    id=0,
    lifetime=rospy.Duration(1.5),
    pose=Pose(Point(0.5, 0.5, 1.45), Quaternion(0, 0, 0, 1)),
    scale=Vector3(1.50, 1.50, 1.50),
    header=Header(frame_id='base_link'),
    color=ColorRGBA(0.0, 1.0, 0.0, 0.9),
    text=text)
    marker_publisher.publish(marker)


def pub_motor(Angle, Speed):
    drive_info = [Angle, Speed]
    drive_info = Int32MultiArray(data = drive_info)
    pub.publish(drive_info)

def start():
    global pub
    rospy.init_node('my_driver')
    pub = rospy.Publisher('xycar_motor_msg', Int32MultiArray, queue_size=1)
    marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=5)

    capture = cv2.VideoCapture("/home/jjong/catkin_ws/src/xycar_simul/src/track-s.mkv")

    rate = rospy.Rate(60)
    Speed = 20
    current = 0
    while True:
        ret , img = capture.read()
        if ret == False:
            break
        now = time.time()
        bev = BirdEyeView(img)
        ldt = LaneDetector(bev)

        Binary_img = image_processing(bev, img)
        info = ldt.slidingWindows(Binary_img)
        final_frame, left_curverad, right_curverad = ldt.drawFitLane(img, Binary_img, info)
        left_curvature, right_curvature= 1/left_curverad, 1/right_curverad

        if info['valid_left_line'] & info['valid_right_line'] :
            final_curvature = WEIGHT*(left_curvature + right_curvature)/2
        elif info['valid_left_line'] :
            final_curvature = WEIGHT*left_curvature
        elif info['valid_right_line'] :
            final_curvature = WEIGHT*right_curvature
        
        if final_curvature >50 :
            final_curvature = 50
        elif final_curvature < -50 :
            final_curvature = -50
        
        #cv2.imshow("roi_frame", roi_frame)
        cv2.imshow("Binary_img", Binary_img)
        #cv2.imshow("warped_frame2",warped_frame2)

        #문자 출력
        cv2.putText(final_frame, "Radius of curvature : " + str(final_curvature), (10,  30), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 255, 0),  2)
#        cv2.imshow('image',final_frame)
        if cv2.waitKey(100) > 0 : break
        print("time : ", now - current)
        print("=============================")
        current = time.time()
        show_text_in_rviz(marker_publisher, "1/curvature : " + str(int(final_curvature)))
        pub_motor((final_curvature), Speed)
        rate.sleep()


if __name__ == '__main__':
    start()
