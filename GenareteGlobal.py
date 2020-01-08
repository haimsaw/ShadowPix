import itertools

from shadoPixToStl import create_stl_global
import numpy
import math
from scipy import signal, optimize
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy
import multiprocessing

# todo any number of images

image_size = 500
# directions = ["-y", "+y", "-x", "+x"]
directions = ["+y", "+x"]
# directions = ["+x"]
num_of_images = len(directions)
light_angle = math.radians(30)


class Telemetry:
    def __init__(self):
        self.iteration_duration = 0
        self.take_step_duration = 0
        self.eval_f_duration = 0
        self.get_lit_map = 0
        self.num_of_mesurments = 0

    def update_messurment(self, start_iteration_time, take_step_time, evel_f_time, end_iteration_time):
        self.iteration_duration += end_iteration_time - start_iteration_time
        self.take_step_duration += take_step_time - start_iteration_time
        self.eval_f_duration += evel_f_time - take_step_time
        self.num_of_mesurments += 1

    def update_get_lit_map(self, duration):
        self.get_lit_map += duration

    def __str__(self):
        return "telemetry iter:{0:03f} step:{1:03f} eval={2:03f} lit_map={3:03f} niter={4}".format(
            self.iteration_duration / self.num_of_mesurments, self.take_step_duration / self.num_of_mesurments,
            self.eval_f_duration / self.num_of_mesurments, self.get_lit_map / self.num_of_mesurments,
            self.num_of_mesurments)


class State:
    def __init__(self):  # todo pass step taker
        self.heightfield = numpy.zeros((image_size, image_size), dtype=int)
        self.lit_maps = {direction: numpy.ones((image_size, image_size), dtype=int) for direction in directions}

    def get_lit_map(self, direction):
        return self.lit_maps[direction]

    def copy(self):
        return copy.deepcopy(self)

    def update_lit_pixel(self, i, j, shadow_height, direction):
        pixel_height = self.heightfield[i][j]
        if shadow_height < pixel_height or shadow_height == 0:
            shadow_height = pixel_height
            self.lit_maps[direction][i][j] = 1
        else:
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
            i = random.randint(0, image_size - 1)
            j = random.randint(0, image_size - 1)
            is_add = random.randint(0, 1) == 0
            new_val = self.heightfield[i][j] + 5 if is_add else self.heightfield[i][j] - 5
            if 0 <= new_val <= 10:
                self.heightfield[i][j] = new_val
                start = time.time()
                self.update_lit_maps(i, j)
                telemetry.update_get_lit_map(time.time() - start)
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
    def __init__(self, images):
        self.gradient_kernel = numpy.asarray([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]])

        self.gaussian_kernel = numpy.asarray([[1, 4, 7, 4, 1],
                                         [4, 16, 26, 16, 4],
                                         [7, 26, 41, 26, 7],
                                         [4, 16, 26, 16, 4],
                                         [1, 4, 7, 4, 1]]) / 273

        self.gaus_grad_kernel = signal.convolve2d(self.gaussian_kernel, self.gradient_kernel, mode="same")

        self.images = images
        self.edges = {direction: signal.convolve2d(images[direction], self.gaus_grad_kernel, mode="same")
                      for direction in directions}

    def val_for_direction(self, lit_map, direction):
        res = 0
        res += StateEvaluator.norma(signal.convolve2d(lit_map, self.gaussian_kernel, mode="same"), self.images[direction])
        res += StateEvaluator.norma(signal.convolve2d(lit_map, self.gaus_grad_kernel, mode="same"), self.edges[direction]) * 1.5
        return res

    def val_for_heightfield(self, heightfield):
        return StateEvaluator.norma(signal.convolve2d(heightfield, self.gradient_kernel, mode="same"),
                     numpy.zeros((image_size, image_size))) * 0.001

    @staticmethod
    def norma(im1, im2):
        return numpy.sum((im1 - im2) ** 2)


def main():
    images = get_images()
    state_evaluator = StateEvaluator(images)
    fig = plt.figure(figsize=(10, 10))
    heightfield = my_optimize(state_evaluator, images, fig)

    save_res(heightfield, images, fig)
    plt.show()
    #  create_stl_global(heightfield)


def save_res(stet, images, fig):
    for i in range(num_of_images):
        direction = directions[i]
        fig.add_subplot(num_of_images, 2, 2 * i + 1)
        plt.imshow(stet.get_lit_map(direction), cmap='gray')

        fig.add_subplot(num_of_images, 2, 2 * i + 2)
        plt.imshow(images[direction], cmap='gray')
    plt.savefig("res.pdf")
    numpy.save("res_heightfield.np", stet.heightfield)


def get_images():
    # images.append(numpy.asarray([[1 if i % 5 == 0 else 0 for i in range(image_size)] for _ in range(image_size)]))
    #  images.append(numpy.random.rand(image_size, image_size))

    get_im = lambda direction: numpy.array(Image.open("img{0}.jpg".format(direction)).
                                           convert("L").crop((0, 0, image_size, image_size))) / 255
    images = {direction: get_im(direction) for direction in directions}
    return images


def f_to_minimize(pool, state_evaluator, state):

    args = [(state.get_lit_map(direction), direction) for direction in directions]

    async_res = pool.starmap_async(state_evaluator.val_for_direction, args)
    res = state_evaluator.val_for_heightfield(state.heightfield)
    res += sum(async_res.get())

    return res


def my_optimize(state_evaluator, images, fig):
    def callback(iteration_num, state, telemetry):
        if iteration_num % 50 == 0:  # todo this

            save_res(state, images, fig)
            print(telemetry)

    '''
    def no_minimizer(fun, x0, args, jac, hess, hessp,
                      bounds, constraints,
                      callback, *options):
        return optimize.OptimizeResult(x=x0, success=True, fun=fun(x0))

    minimizer_kwargs = {"method": no_minimizer}

    # todo constant temp
    #res = optimize.basinhopping(f, x0=inital_guss, take_step=mytakestep, disp=True, minimizer_kwargs=minimizer_kwargs, niter=10**7, niter_success=1000).x
    '''

    res = simulated_annealing(state_evaluator, 10 ** 7, 100000, State(), callback)
    return res


def simulated_annealing(state_evaluator, niter, niter_success, state, callback):
    telemetry = Telemetry()
    pool = multiprocessing.Pool(processes=num_of_images)

    val = f_to_minimize(pool, state_evaluator, state)
    global_min_val = val
    global_min_state = state
    success_iter = 0

    for iteration_num in range(niter):
        if niter_success < success_iter:
            break
        start_iteration_time = time.time()
        temp = niter - iteration_num - 1  # todo, check this
        candidate_state = state.copy().take_step(telemetry)
        take_step_time = time.time()
        candidate_val = f_to_minimize(pool, state_evaluator, candidate_state)
        evel_f_time = time.time()

        if candidate_val < val or random.random() <= math.exp((val - candidate_val) / temp):
            state = candidate_state
            val = candidate_val
        print("iteration={0} val={1} global val={2} success_iter={3}".
              format(iteration_num, int(val), int(global_min_val), success_iter))

        if val < global_min_val:
            global_min_val = val
            global_min_state = state
            success_iter = 0

        else:
            success_iter += 1

        telemetry.update_messurment(start_iteration_time, take_step_time, evel_f_time, time.time())

        callback(iteration_num, state, telemetry)
    return global_min_state


if __name__ == "__main__":
    main()
