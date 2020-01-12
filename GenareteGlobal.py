import itertools

from shadoPixToStl import create_stl_global
from math import radians, exp 
from scipy.signal import convolve2d
from random import randint, random
from PIL import Image
import matplotlib.pyplot as plt
from time import time, sleep
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import numpy
import os

# todo any number of images

image_size = 400  # max in article 200/0.5=400

num_of_images = 4
num_of_cores = cpu_count()
directions = ["+x", "+y", "-x", "-y"][:num_of_images]
light_angle = radians(30)


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

        return "\ttelemetry iter:{:03f} step:{:03f} get_accepted_steps_time={:03f} niter={}".format(
            self.iteration_duration / self.num_of_mesurments, self.take_step_duration / self.num_of_mesurments,
            self.get_accepted_steps_time / self.num_of_mesurments, self.num_of_mesurments)


class State:
    def __init__(self):  # todo pass step taker
        self.heightfield = numpy.zeros((image_size, image_size), dtype=int)
        self.lit_maps = {direction: numpy.ones((image_size, image_size), dtype=int) for direction in directions}
        self.val = None

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
        # todo this can be improved - find shadow_height above i,j and iterate for pixel_height
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
                if abs(existine_step["i"] - step["i"]) <= step_radios or abs(existine_step["j"] - step["j"]) <= step_radios:
                    should_accept = False
                    break
            if should_accept:
                steps.append(step)
        return steps

    def get_val(self, state_evaluator):
        if self.val is not None:
            return self.val

        args = [(self.get_lit_map(direction), direction) for direction in directions]
        self.val = sum(itertools.chain(itertools.starmap(state_evaluator.val_for_direction, args),
                                      [state_evaluator.val_for_heightfield(self.heightfield)]))
        return self.val

    def apply_step(self, step):
        self.heightfield[step["i"]][step["j"]] = step["height"]
        self.update_lit_maps(step["i"], step["j"])
        self.val = None


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


def main():
    images = get_images()
    state_evaluator = StateEvaluator(images)
    heightfield = my_optimize(state_evaluator, images)

    save_res(heightfield, images)
    plt.show()
    create_stl_global(heightfield)


def save_res(state, images):
    fig, axs = plt.subplots(nrows=num_of_images, ncols=2, figsize=(10, 10),  # todo reuse fig
                            subplot_kw={'xticks': [], 'yticks': []})

    for i in range(num_of_images):
        direction = directions[i]
        axs[i, 0].imshow(state.get_lit_map(direction), cmap='gray')
        axs[i, 1].imshow(images[direction], cmap='gray')
        axs[i, 0].set_title(direction)
    plt.tight_layout()
    plt.savefig("res.pdf")
    plt.close(fig)
    numpy.save("res_heightfield.np", state.heightfield)


def get_images():
    # images.append(numpy.asarray([[1 if i % 5 == 0 else 0 for i in range(image_size)] for _ in range(image_size)]))
    #  images.append(numpy.rand(image_size, image_size))

    get_im = lambda direction: numpy.array(Image.open("img{0}.jpg".format(direction)).
                                           convert("L").resize((image_size, image_size), )) / 255
    images = {direction: get_im(direction) for direction in directions}
    return images


'''
def f_to_minimize_concurrent(pool, state_evaluator, state, telemetry):
    res = calc_times = 0
    args = [(state.get_lit_map(direction), direction) for direction in directions]

    async_res = pool.starmap_async(state_evaluator.val_for_direction, args)
    evaluations = itertools.chain([state_evaluator.val_for_heightfield(state.heightfield)], async_res.get())

    for evaluation in evaluations:
        res += evaluation[0]
        calc_times += evaluation[1]

    if telemetry is not None:
        telemetry.update_total_eval_f_duration(calc_times)
    return res

'''


def my_optimize(state_evaluator, images):
    def callback(iteration_num, state, telemetry):
        if iteration_num % 50 == 0:  # todo this
            save_res(state, images)
            print(telemetry)

    with Pool(processes=num_of_cores) as pool:
        return simulated_annealing(state_evaluator, 10 ** 7, 10 ** 5, State(), callback, pool, 1)


def is_step_accepted(state, step, state_evaluator, temp):
    candidate = state.copy()
    candidate.apply_step(step)
    metopolis_test = exp((state.get_val(state_evaluator) - candidate.get_val(state_evaluator)) / temp)
    return random() <= metopolis_test, step


def simulated_annealing(state_evaluator, niter, niter_success, state, callback, pool, initial_temperature):

    def get_accepted_steps(candidate_steps, state):
        args = [(state, step, state_evaluator, temp) for step in candidate_steps]
        copy = state.copy()

        for res, step in pool.starmap_async(is_step_accepted, args).get():
            if res:
                copy.apply_step(step)
        return copy

    def get_accepted_steps_non_concurrent(candidate_steps, state):
        args = [(state, step, state_evaluator, temp) for step in candidate_steps]
        copy = state.copy()

        for res, step in itertools.starmap(is_step_accepted, args):
            if res:
                copy.apply_step(step)
        return copy

    global_min_state = state
    min_time_at_top = 0
    telemetry = Telemetry()

    for iteration_num in range(niter):
        if niter_success < min_time_at_top:
            break
        start_iteration_time = time()
        temp = (niter - iteration_num)/niter*initial_temperature
        candidate_steps = state.get_steps(num_of_cores)

        take_step_time = time()
        state = get_accepted_steps_non_concurrent(candidate_steps, state)

        get_accepted_steps_time = time()

        print("iteration={} val={} min={} time_at_top={} temp={:.2f}".
              format(iteration_num, int(state.get_val(state_evaluator)), int(global_min_state.get_val(state_evaluator)), min_time_at_top, temp))

        if state.get_val(state_evaluator) < global_min_state.get_val(state_evaluator):
            global_min_state = state
            min_time_at_top = 0

        else:
            min_time_at_top += 1

        telemetry.update_messurment(start_iteration_time, take_step_time, get_accepted_steps_time, time())

        callback(iteration_num, global_min_state, telemetry)
    return global_min_state


if __name__ == "__main__":
    main()
