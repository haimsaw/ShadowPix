
from shadowPixToStl import create_stl_global
from math import radians, exp 
from scipy.signal import convolve2d
from random import randint, random
from PIL import Image
from time import time
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from itertools import chain, starmap

import numpy
import matplotlib.pyplot as plt
import argparse


class Telemetry:
    def __init__(self):
        self.iteration_duration = 0
        self.take_step_duration = 0
        self.get_accepted_steps_time = 0
        self.num_of_mesurments = 0

    def update_messurment(self, start_iteration_time, take_step_time, get_accepted_steps_time, end_iteration_time):
        self.iteration_duration += end_iteration_time - start_iteration_time
        self.take_step_duration += take_step_time - start_iteration_time
        self.get_accepted_steps_time += get_accepted_steps_time - take_step_time
        self.num_of_mesurments += 1

    def __str__(self):
        return "\ttimings iter:{:03f} step:{:03f} get_accepted_steps_time={:03f} niter={}".format(
            self.iteration_duration / self.num_of_mesurments, self.take_step_duration / self.num_of_mesurments,
            self.get_accepted_steps_time / self.num_of_mesurments, self.num_of_mesurments)


class StateEvaluator:
    gradient_kernel = numpy.asarray([[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]])

    gaussian_kernel = numpy.asarray([[1, 4, 7, 4, 1],
                                          [4, 16, 26, 16, 4],
                                          [7, 26, 41, 26, 7],
                                          [4, 16, 26, 16, 4],
                                          [1, 4, 7, 4, 1]]) / 273

    gaus_grad_kernel = convolve2d(gaussian_kernel, gradient_kernel, mode="same")

    def __init__(self, images):
        self.images = images
        self.edges = {direction: convolve2d(self.images[direction], self.gaus_grad_kernel, mode="same")
                      for direction in directions}

    def val_for_direction(self, lit_map, direction):
        res = 0
        res += StateEvaluator.norma(convolve2d(lit_map, self.gaussian_kernel, mode="same"), self.images[direction])
        res += StateEvaluator.norma(convolve2d(lit_map, self.gaus_grad_kernel, mode="same"), self.edges[direction]) * 1.5
        return res

    def val_for_heightfield(self, heightfield):
        return StateEvaluator.norma(convolve2d(heightfield, self.gradient_kernel, mode="same")) * 0.001

    @staticmethod
    def norma(im1, im2=None):
        if im2 is not None:
            return numpy.sum((im1 - im2) ** 2)
        return numpy.sum(im1 ** 2)


class State:
    def __init__(self, images):
        self.heightfield = numpy.zeros((image_size, image_size), dtype=int)
        self.lit_maps = {direction: numpy.ones((image_size, image_size), dtype=int) for direction in directions}
        self.val = None
        self.evaluator = StateEvaluator(images)

    def get_lit_map(self, direction):
        return self.lit_maps[direction]

    def copy(self):
        return deepcopy(self)

    def update_lit_pixel(self, i, j, shadow_height, direction):
        pixel_height = self.heightfield[i][j]
        if shadow_height < pixel_height or shadow_height == 0:
            # lit
            shadow_height = pixel_height
            self.lit_maps[direction][i][j] = 1
        else:
            # shadowed
            self.lit_maps[direction][i][j] = 0

        return shadow_height - 1 if shadow_height > 0 else 0

    def update_lit_maps(self, i, j):
        # this can be improved - find shadow_height above i,j and iterate for pixel_height
        for direction in directions:
            if direction == "-y":
                next_shadow_height = 0
                for iter_i in range(image_size - 1, -1, -1):
                    next_shadow_height = self.update_lit_pixel(iter_i, j, next_shadow_height, direction)

            elif direction == "+y":
                next_shadow_height = 0
                for iter_i in range(image_size):
                    next_shadow_height = self.update_lit_pixel(iter_i, j, next_shadow_height, direction)

            elif direction == "-x":
                next_shadow_height = 0
                for iter_j in range(image_size - 1, -1, -1):
                    next_shadow_height = self.update_lit_pixel(i, iter_j, next_shadow_height, direction)

            elif direction == "+x":
                next_shadow_height = 0
                for iter_j in range(image_size):
                    next_shadow_height = self.update_lit_pixel(i, iter_j, next_shadow_height, direction)

    def get_step_random(self):
        while True:
            i = randint(0, image_size - 1)
            j = randint(0, image_size - 1)
            change = randint(-5, 5)
            new_val = self.heightfield[i][j] + change
            if change != 0 and 0 <= new_val <= 10:
                return {"i": i, "j": j, "height": new_val}

    def get_steps(self, num_of_steps):
        steps = []
        step_radios = 20
        while len(steps) < num_of_steps:
            step = self.get_step_random()
            should_accept = True
            for existine_step in steps:
                if abs(existine_step["i"] - step["i"]) <= step_radios or abs(
                        existine_step["j"] - step["j"]) <= step_radios:
                    should_accept = False
                    break
            if should_accept:
                steps.append(step)
        return steps

    def get_val(self,):
        if self.val is not None:
            return self.val
        args = [(self.get_lit_map(direction), direction) for direction in directions]
        self.val = sum(chain(starmap(self.evaluator.val_for_direction, args),
                             [self.evaluator.val_for_heightfield(self.heightfield)]))
        return self.val

    def apply_step(self, step):
        self.heightfield[step["i"]][step["j"]] = step["height"]
        self.update_lit_maps(step["i"], step["j"])
        self.val = None


def main():
    print("ShadowPix global nImages={}, directions={}, size={}px, concurrency={} nIterations={}".format(
        num_of_images, directions, image_size, is_concurrency, num_iterations))

    images = get_images()
    final_state = my_optimize(images)

    save_res(final_state, images)
    create_stl_global(final_state.heightfield)


def parse_args():
    global num_of_cores, image_size, num_of_images, directions, is_concurrency, is_verbose, stop_after, num_iterations, temp_incline_steps
    parser = argparse.ArgumentParser(description='Create shadowPix -global method')

    parser.add_argument('--nImages', '-i', default=4, type=int, help='num of images to process, default=4, 0<n<=4')
    parser.add_argument('--nPixels', '-p', default=300, type=int, help='ShadowPix size in pixels, default=300')
    parser.add_argument('--lightAngle', '-a', default=30, type=int,
                        help='illumination angle in degrees, default=30, 0<n<90')

    parser.add_argument('--verbose', '-v', action='store_true', help='should print iteration data, default=False')
    parser.add_argument('--concurrency', '-c', action='store_true',
                        help='should use concurrency optimization, default=False, not recommended on windows')

    parser.add_argument('--nSteps', '-it', default=10 ** 7, type=int,
                        help='max num of simulated_annealing steps, default=10 ** 7, each iteration has steps according to num of cores')
    parser.add_argument('--stopAfter', '-s', default=10000, type=int,
                        help='finish optimization if the global minimum candidate remains the same for this number of steps, default=10000')
    parser.add_argument('--tempIncline', '-t', default=10000, type=int,
                        help='reduce temp incline if the global minimum candidate remains the same for this number of steps, default=10000')

    args = parser.parse_args()

    num_of_cores = cpu_count()
    image_size = args.nPixels
    num_of_images = args.nImages
    directions = ["+x", "+y", "-x", "-y"][:num_of_images]
    light_angle = radians(args.lightAngle)
    is_concurrency = bool(args.concurrency)
    is_verbose = bool(args.verbose)
    stop_after = int(args.stopAfter / num_of_cores)
    num_iterations = int(args.nSteps / num_of_cores)
    temp_incline_steps = int(args.tempIncline / num_of_cores)


def save_res(state, images):

    fig, axs = plt.subplots(nrows=num_of_images+1, ncols=2, figsize=(10, 10),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i in range(num_of_images):
        direction = directions[i]
        axs[i, 0].imshow(state.get_lit_map(direction), cmap='gray')
        axs[i, 1].imshow(images[direction], cmap='gray')
        axs[i, 0].set_title(direction)
    axs[num_of_images, 0].imshow(state.heightfield, cmap='gray_r')
    axs[num_of_images, 0].set_title("heightfield")

    plt.tight_layout()
    plt.savefig("res_images.pdf")
    plt.close(fig)
    numpy.save("res_heightfield", state.heightfield)


def get_images():
    get_im = lambda direction: numpy.array(Image.open("img{0}.jpg".format(direction)).
                                           convert("L").resize((image_size, image_size), )) / 255
    images = {direction: get_im(direction) for direction in directions}
    return images


def is_step_accepted(state, step, temp):

    candidate = state.copy()
    candidate.apply_step(step)
    if temp == 0:
        return state.get_val() > candidate.get_val(), step
    else:
        return random() <= exp((state.get_val() - candidate.get_val()) / temp), step


def my_optimize(images):
    def callback(iteration_num, state, telemetry):
        if iteration_num % 1000 == 0:
            save_res(state, images)
            if is_verbose:
                print(telemetry)

    state = State(images)

    if is_concurrency:
        with Pool(processes=num_of_cores) as pool:
            return simulated_annealing(num_iterations, stop_after, state, callback, pool, 1)
    else:
        return simulated_annealing(num_iterations, stop_after, state, callback, None, 1)


def simulated_annealing(niter, niter_success, state, callback, pool, initial_temperature):

    def accept_and_apply_steps():
        args = [(state, step, temp) for step in candidate_steps]
        copy = state.copy()
        iter = pool.starmap_async(is_step_accepted, args).get() if is_concurrency and pool is not None\
            else starmap(is_step_accepted, args)

        for res, step in iter:
            if res:
                copy.apply_step(step)
        return copy

    global_min_state = state
    min_time_at_top = 0
    telemetry = Telemetry()
    temp_incline = initial_temperature

    for iteration_num in range(niter):

        if niter_success < min_time_at_top:
            print("could not fined better min for {} iterations. ".format(min_time_at_top))
            break

        elif temp_incline == initial_temperature and min_time_at_top == temp_incline_steps:
            temp_incline = initial_temperature/2
            min_time_at_top = 0
            print("reducing temp incline to {}".format(temp_incline))

        elif temp_incline == initial_temperature/2 and min_time_at_top == temp_incline_steps:
            temp_incline = 0
            min_time_at_top = 0
            print("reducing temp incline to {}".format(temp_incline))

        start_iteration_time = time()

        temp = (niter - iteration_num)/niter*temp_incline
        candidate_steps = state.get_steps(num_of_cores)

        take_step_time = time()
        state = accept_and_apply_steps()

        get_accepted_steps_time = time()

        if state.get_val() < global_min_state.get_val():
            global_min_state = state
            min_time_at_top = 0

        else:
            min_time_at_top += 1

        if is_verbose:
            print("iteration={} val={:.2f} min={:.2f} global_min_time={} temp={:.2f}".
                  format(iteration_num, state.get_val(),
                         global_min_state.get_val(), min_time_at_top, temp))

        telemetry.update_messurment(start_iteration_time, take_step_time, get_accepted_steps_time, time())

        callback(iteration_num, global_min_state, telemetry)
    return global_min_state


parse_args()

if __name__ == "__main__":
    main()

