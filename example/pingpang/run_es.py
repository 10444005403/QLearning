#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 xuekun.zhuang <zhuangxuekun@imdada.cn>
# Licensed under the Dada tech.co.ltd - http://www.imdada.cn
import multiprocessing
import numpy as np
import cPickle as pickle
import gym
import time
import math
import logging
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
from gym.envs.registration import logger


logger.setLevel(logging.ERROR)


class Player(object):

    MOVE_DOWN = 2
    MOVE_UP = 3

    def __init__(self):
        self.hidden_neuron_num = 4
        self.input_dim = 16*16
        self.model = dict()
        self.model['W1'] = np.random.randn(self.hidden_neuron_num, self.input_dim) / \
                           np.sqrt(self.hidden_neuron_num)
        self.model['W2'] = np.random.randn(self.hidden_neuron_num) / np.sqrt(self.hidden_neuron_num)

        self.env = gym.make("Pong-v0")
        self.need_render = False
        self.game_action_counter = 0
        self.game_counter = 0
        self.game_round_counter = 0
        self.chromosome_size = self.input_dim*self.hidden_neuron_num + self.hidden_neuron_num

    def play(self):
        """play main steps"""
        observation = self.env.reset()
        prev_x = None
        while True:

            if self.need_render:
                self.env.render()

            # preprocess the observation, set input to network to be difference image
            cur_x = self.prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.input_dim)
            prev_x = cur_x

            # forward the policy network and sample an action from the returned probability
            aprob, h = self.policy_forward(x)

            # roll the dice !
            action = self.MOVE_DOWN if np.random.uniform() < aprob else self.MOVE_UP

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action)

            # set counter
            self.game_action_counter += 1
            if reward != 0:
                # a game finished
                self.game_counter += 1

                # transform chromosome
                # print self.translate_w([self.model['W1'], self.model['W2']])
                break

        return reward

    @staticmethod
    def translate_w(w_list):
        w_list = [w.ravel() for w in w_list]
        return np.hstack(w_list)

    @staticmethod
    def translate_chromosome(chromosome, w_sizes):
        w_trans = []
        start_index = 0
        for row, col in w_sizes:
            len = row * col
            if row != 1:
                w = np.reshape(chromosome[start_index:start_index+len], (row, col))
            else:
                w = chromosome[start_index:start_index + len]
            w_trans.append(w)
            start_index += len
        return w_trans

    @staticmethod
    def sigmoid(x):
        p = 0
        try:
            p = 1.0 / (1.0 + math.exp(-x))
        except:
            pass
        return p

    @staticmethod
    def prepro(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195]       # crop
        I = I[::10, ::10, 0]  # downsample by factor of 2
        I[I == 144] = 0     # erase background (background type 1)
        I[I == 109] = 0     # erase background (background type 2)
        I[I != 0] = 1       # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0        # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h         # return probability of taking action 2, and hidden state

    def load_model(self, model_file):
        self.model = pickle.load(open(model_file, 'rb'))


def init():
    global player
    player = Player()


def get_ind_fitness(ind):
    w_trans = player.translate_chromosome(ind, [(player.hidden_neuron_num,
                                                 player.input_dim),
                                                (1, player.hidden_neuron_num)])
    player.model['W1'] = w_trans[0]
    player.model['W2'] = w_trans[1]
    return player.play(),


class Evolver(object):

    def __init__(self):
        self.pop_size = 100
        self.dim_size = 0
        self.generation_num = 0
        self.player = Player()

    def set_param(self, generation_num):
        self.dim_size = self.player.chromosome_size
        self.generation_num = generation_num

    def get_fitness(self, population):
        fitnesses = []
        for ind in population:
            w_trans = self.player.translate_chromosome(ind,
                        [(self.player.hidden_neuron_num, self.player.input_dim),
                        (1, self.player.hidden_neuron_num)])
            self.player.model['W1'] = w_trans[0]
            self.player.model['W2'] = w_trans[1]
            fitnesses.append((self.player.play(),))
        return fitnesses

    @staticmethod
    def get_fitness_multi_process(population):
        pool = multiprocessing.Pool(processes=10, initializer=init)
        fitnesses = []
        for ind in population:
            fitnesses.append(pool.apply_async(get_ind_fitness, (ind,)))
        pool.close()
        pool.join()
        fitnesses = [f.get() for f in fitnesses]
        return fitnesses

    def run(self):

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("win", lambda x: (np.array([i[0] for i in np.asarray(x)]) > 0).sum())

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max", "win"
        print "dim_size: %d " % self.dim_size

        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", benchmarks.rastrigin)

        np.random.seed(64)

        strategy = cma.Strategy(centroid=[1.0]*self.dim_size, sigma=1.0,
                                lambda_=self.pop_size, stats=stats)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        halloffame = tools.HallOfFame(1)

        for gen in range(self.generation_num):

            recorder = open('cmaes.log', 'w')

            # Generate a new population
            population = toolbox.generate()

            # Evaluate the individuals
            fitnesses = self.get_fitness_multi_process(population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Update the strategy with the evaluated individuals
            toolbox.update(population)

            # Update the hall of fame and the statistics with the
            # currently evaluated population
            halloffame.update(population)
            record = stats.compile(population)
            logbook.record(evals=len(population), gen=gen, **record)
            print(logbook.stream)
            recorder.write("PPP")
            if gen % 100 == 0:
                pickle.dump(self.player.model, open('cmaes.pkl', 'wb'))

            recorder.close()


if __name__ == "__main__":

    pass

    # p_1 = Player()
    # p_1.load_model("./cmaes.pkl")
    # reward = p_1.play()
    # print reward

    runner = Evolver()
    runner.set_param(1000000)
    runner.run()
