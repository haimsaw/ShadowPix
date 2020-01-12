import itertools

from shadoPixToStl import create_stl_global
from math import radians, exp 
from scipy.signal import convolve2d
from random import randint, random
from PIL import Image
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
import multiprocessing
import numpy


# todo any number of images

image_size = 300  # max in article 200/0.5=400

num_of_images = 4
directions = ["+x", "+y", "-x", "-y"][:num_of_images]
light_angle = radians(30)


class Telemetry:
    def __init__(self):
        self.iteration_duration = 0
        self.take_step_duration = 0
        self.eval_f_duration = 0
        self.get_lit_map = 0
        self.num_of_mesurments = 0
        self.total_eval_f_duration = 0

    def update_messurment(self, start_iteration_time, take_step_time, evel_f_time, end_iteration_time):
        self.iteration_duration += end_iteration_time - start_iteration_time
        self.take_step_duration += take_step_time - start_iteration_time
        self.eval_f_duration += evel_f_time - take_step_time
        self.num_of_mesurments += 1

    def update_get_lit_map(self, duration):
        self.get_lit_map += duration

    def update_total_eval_f_duration(self, duration):
        self.total_eval_f_duration += duration

    def __str__(self):

        return "\ttelemetry iter:{:03f} step:{:03f} eval={:03f} total_eval={:03f} lit_map={:03f} niter={}".format(
            self.iteration_duration / self.num_of_mesurments, self.take_step_duration / self.num_of_mesurments,
            self.eval_f_duration / self.num_of_mesurments, self.total_eval_f_duration / self.num_of_mesurments,
            self.get_lit_map / self.num_of_mesurments, self.num_of_mesurments)


class State:
    def __init__(self):  # todo pass step taker
        self.heightfield = numpy.zeros((image_size, image_size), dtype=int)
        self.lit_maps = {direction: numpy.ones((image_size, image_size), dtype=int) for direction in directions}

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

    def take_step_random(self, telemetry):
        while True:
            i = randint(0, image_size - 1)
            j = randint(0, image_size - 1)
            change = randint(-5, 5)
            new_val = self.heightfield[i][j] + change
            if change != 0 and 0 <= new_val <= 10:
                self.heightfield[i][j] = new_val
                start = time()
                self.update_lit_maps(i, j)
                telemetry.update_get_lit_map(time() - start)
                break

    def take_step_all(self, telemetry):
        pass  # todo
        '''
        class MyTakeStepAll:
            def __init__(self):
                self.y = self.x = image_size - 1
                self.is_add = False
        
            def __call__(self, state):
                while True:
                    self.x = (self.x + 1) % image_size
                    if self.x == 0:
                        self.y = (self.y + 1) % image_size
                    if self.x == self.y == 0:
                        self.is_add = not self.is_add
        
                    new_val = state[self.x][self.y] + 5 if self.is_add else state[self.x][self.y] - 5
                    if 0 <= new_val <= 10:
                        new_state = numpy.copy(state)
                        new_state[self.x][self.y] = new_val
                        break
                return new_state
        
            def get_niter_success(self):
                return image_size*image_size*2
        '''

    def take_step(self, telemetry):
        self.take_step_random(telemetry)
        return self


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
        start = time()
        res = 0
        res += StateEvaluator.norma(convolve2d(lit_map, self.gaussian_kernel, mode="same"), self.images[direction])
        res += StateEvaluator.norma(convolve2d(lit_map, self.gaus_grad_kernel, mode="same"), self.edges[direction]) * 1.5
        return res, time() - start

    def val_for_heightfield(self, heightfield):
        start = time()
        return StateEvaluator.norma(convolve2d(heightfield, self.gradient_kernel, mode="same")) * 0.001, \
               time() - start

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


def f_to_minimize_non_concurrent(pool, state_evaluator, state, telemetry):
    res = calc_times = 0
    args = [(state.get_lit_map(direction), direction) for direction in directions]
    evaluations = itertools.chain(itertools.starmap(state_evaluator.val_for_direction, args),
                                  [state_evaluator.val_for_heightfield(state.heightfield)])

    for evaluation in evaluations:
        res += evaluation[0]
        calc_times += evaluation[1]
    if telemetry is not None:
        telemetry.update_total_eval_f_duration(calc_times)
    return res


def my_optimize(state_evaluator, images):
    def callback(iteration_num, state, telemetry):
        if iteration_num % 1000 == 0:  # todo this
            save_res(state, images)
            print(telemetry)

    with multiprocessing.Pool(processes=num_of_images) as pool:
        return simulated_annealing(state_evaluator, f_to_minimize_concurrent, 10 ** 7, 10 ** 5, State(),
                                   callback, pool, 1)


def simulated_annealing(state_evaluator, f, niter, niter_success, state, callback, pool, initial_temperature):

    val = f(pool, state_evaluator, state, None)
    global_min_val = val
    global_min_state = state
    min_time_at_top = 0
    telemetry = Telemetry()

    for iteration_num in range(niter):
        if niter_success < min_time_at_top:
            break
        start_iteration_time = time()
        temp = (niter - iteration_num)/niter*initial_temperature
        candidate_state = state.copy().take_step(telemetry)
        take_step_time = time()
        candidate_val = f(pool, state_evaluator, candidate_state, telemetry)
        evel_f_time = time()
        metopolis_test = exp((val - candidate_val) / temp)

        print("iteration={} val={} candidate_val={} metopolis={:.2f} min={} time_at_top={} temp={:.2f}".
              format(iteration_num, int(val), int(candidate_val), metopolis_test, int(global_min_val), min_time_at_top, temp))

        if random() <= metopolis_test:
            state = candidate_state
            val = candidate_val

        if val < global_min_val:
            global_min_val = val
            global_min_state = state
            min_time_at_top = 0

        else:
            min_time_at_top += 1

        telemetry.update_messurment(start_iteration_time, take_step_time, evel_f_time, time())

        callback(iteration_num, global_min_state, telemetry)
    return global_min_state


if __name__ == "__main__":
    main()
