#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 xuekun.zhuang <zhuangxuekun@imdada.cn>

import numpy as np
import random


class Agent(object):

    def __init__(self):
        self._gama = 0.8
        self._episode = 10
        self.R = Reward.R
        self.Q = Brain.Q

    def set_episode(self, episode):
        self._episode = episode

    def train(self):

        for i in xrange(self._episode):
            # 随机选取一个初始状态
            select_status = random.randint(0, 5)

            # 选择出跳跃状态
            indexes = self.get_skip_status(select_status)
            skip_status = indexes[random.randint(0, len(indexes) - 1)]

            # 获取跳跃之后的Q矩阵状态
            max_q = self.get_max_q(skip_status)

            # 更新Q矩阵
            self.Q[select_status, skip_status] = self.R[select_status, skip_status] + self._gama*max_q

            # print "select_status:%d\tskip_status:%d\tmax_q:%d" % (select_status, skip_status, max_q)
            # print self.Q
            # print "-"*64

        print "Train done !"

    def test(self, start_status):
        solutiton = [start_status]
        while True:
            skip_status = np.where(self.Q[start_status] == self.Q[start_status].max())[1][0]
            if start_status == skip_status:
                break
            solutiton.append(skip_status)
            start_status = skip_status
        return solutiton

    def get_skip_status(self, select_status):
        indexes = np.where(self.R[select_status, :] >= 0)[1]
        return indexes

    def get_max_q(self, skip_status):
        indexes = self.get_skip_status(skip_status)
        return np.max(self.Q[skip_status, indexes])


class Reward(object):

    R = np.mat([
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 0, -1, -1, 0, 100]
    ])


class Brain(object):

    Q = np.mat(np.zeros((6, 6)))


if __name__ == "__main__":

    agent = Agent()
    agent.set_episode(10000)
    agent.train()

    print agent.Q

    solution = agent.test(3)

    print "->".join(map(str, solution))
