# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

class BirdEyeView() :
    def __init__(self, img) :
        self.__img = img
        self.img_h = self.__img.shape[0]
        self.img_w = self.__img.shape[1]
        self.__left = [-50, self.img_h]#[150,720]
        self.__right = [self.img_w+150 , self.img_h]#[1250,720]
        self.__left_top = [195, 280] #[245, 251]#[190, 283]
        self.__right_top = [460, 280] #[406,251]#[470,283] ## 화면상에서 위쪽에 있는 거라 탑이라고 표시했음 헷갈리지 말것
       
        self.__src = np.float32([self.__left, self.__left_top, self.__right_top, self.__right]) ## 원본이미지의 warping 포인트
        self.__dst = np.float32([[100,480] , [100,0] , [540, 0],[540,480]]) ## 결과 이미지에서 src가 매칭될 점들 [200,720],[200,0],[980,0],[980,720]

    def setROI(self,frame) :
        self.__roi = np.array([self.__src]).astype(np.int32)
        return cv2.polylines(frame, np.int32(self.__roi),True,(255,0,0),10)## 10 두께로 파란선 그림

    def warpPerspect(self,frame) :
        y = frame.shape[0]
        x = frame.shape[1]
        M = cv2.getPerspectiveTransform(self.__src,self.__dst) ## 시점변환 메트릭스 얻어옴.
        return cv2.warpPerspective(frame, M,(x,y), flags=cv2.INTER_LINEAR) ## 버드아이뷰로 전환

    @property
    def src(self):
        return self.__src

    @property
    def dst(self):
        return self.__dst
   
    @property
    def img(self):
        return self.__img