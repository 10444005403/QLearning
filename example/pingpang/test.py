#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 xuekun.zhuang <zhuangxuekun@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn


# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

import numpy as np

a = np.random.randn(6, 3)
print a
print a[::2]

