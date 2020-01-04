from shadoPixToStl import create_stl_global
import numpy
import math
from scipy import signal, optimize
import random
from PIL import Image
import matplotlib.pyplot as plt

# todo any number of images

image_size = 500
# directions = ["-y", "+y", "-x", "+x"]
directions = ["+y", "+x"]
# directions = ["+x"]
num_of_images = len(directions)
light_angle = math.radians(30)



def main():
    images = get_images()
    func = get_f_to_minimize(images)
    fig = plt.figure(figsize=(10, 10))
    heightfield = my_optimize(func, images, fig)

    save_res(heightfield, images, fig)
    plt.show()
    #  create_stl_global(heightfield)


def save_res(heightfield, images, fig):
    for i in range(num_of_images):
        fig.add_subplot(num_of_images, 2, 2 * i + 1)
        plt.imshow(is_lit_all(i, heightfield), cmap='gray')

        fig.add_subplot(num_of_images, 2, 2 * i + 2)
        plt.imshow(images[i], cmap='gray')
    plt.savefig("res.pdf")
    numpy.save("res_heightfield.np", heightfield)


def get_images():
    images = []
    for i in range(num_of_images):
        #images.append(numpy.asarray([[1 if i % 5 == 0 else 0 for i in range(image_size)] for _ in range(image_size)]))
        #  images.append(numpy.random.rand(image_size, image_size))
        images.append(numpy.array(Image.open("img{0}.jpg" .format(i)).
                                  convert("L").crop((0, 0, image_size, image_size)))/255)

    return images


def get_f_to_minimize(images):
    edges = []
    gradient_kernel = numpy.asarray([[0, 1, 0],
                                    [1, -4, 1],
                                    [0, 1, 0]])

    gaussian_kernel = numpy.asarray([[1, 4, 7, 4, 1],
                                    [4, 16, 26, 16, 4],
                                    [7, 26, 41, 26, 7],
                                    [4, 16, 26, 16, 4],
                                    [1, 4, 7, 4, 1]]) / 273

    gaus_grad_kernel = signal.convolve2d(gaussian_kernel, gradient_kernel, mode="same")
    norma = lambda im1, im2: numpy.sum((im1-im2)**2)

    for image in images:
        edges.append(signal.convolve2d(image, gaus_grad_kernel, mode="same"))

    def f(heitfield):
        # print("eveluating f")
        res = 0
        # heitfield = numpy.reshape(heitfield, (image_size, image_size))
        for i in range(num_of_images):
            lit_map = is_lit_all(i, heitfield)
            res += norma(signal.convolve2d(lit_map, gaussian_kernel, mode="same"), images[i])

            res += norma(signal.convolve2d(lit_map, gaus_grad_kernel, mode="same"), edges[i])*1.5

        res += norma(signal.convolve2d(heitfield, gradient_kernel, mode="same"),
                     numpy.zeros((image_size, image_size)))*0.001
        return res

    return f


def is_lit_all(direction_num, heightfield):
    direction = directions[direction_num]

    res = numpy.zeros((image_size, image_size))

    def update_pixel(i, j, shadow_height):
        pixel_height = heightfield[i][j]
        if shadow_height < pixel_height or shadow_height == 0:
            shadow_height = pixel_height
            res[i][j] = 1
        else:
            res[i][j] = 0

        return shadow_height - 1 if shadow_height > 0 else 0

    if direction == "-y":
        for j in range(image_size):
            next_shadow_height = 0
            for i in range(image_size - 1, -1, -1):
                next_shadow_height = update_pixel(i, j, next_shadow_height)

    elif direction == "+y":
        for j in range(image_size):
            next_shadow_height = 0
            for i in range(image_size):
                next_shadow_height = update_pixel(i, j, next_shadow_height)

    elif direction == "-x":
        for i in range(image_size):
            next_shadow_height = 0
            for j in range(image_size-1, -1, -1):
                next_shadow_height = update_pixel(i, j, next_shadow_height)

    elif direction == "+x":
        for i in range(image_size):
            next_shadow_height = 0
            for j in range(image_size):
                next_shadow_height = update_pixel(i, j, next_shadow_height)

    return res


'''
class mytakestep:
    def __init__(self):
        self.low = []
        self.heigh = []
        self.mid = []

    def __init__(self, low, heigh, mid):
        pass

    def __call__(self, *args, **kwargs):
        is_add = random.randint(0, 1) == 0 or len(self.low) == image_size**2
        arr1 = self.mid
        arr2 = self.low if is_add else self.heigh

        size1 = len(arr1)
        size2 = len(arr2)
        if random.random() < size1/(size1+size2):
            index = random.randint(size1)
            arr
        else:
            pass



'''

def mytakestep(x):
    # x = numpy.reshape(x, (image_size, image_size))

    while True:  # TODO save (i,j)'s in lists according to their h val and choose from that
        i = random.randint(0, image_size-1)
        j = random.randint(0, image_size-1)
        is_add = random.randint(0, 1) == 0
        new_val = x[i][j] + 5 if is_add else x[i][j] - 5
        if 0 <= new_val <= 10:
            x[i][j] = new_val
            return x


def my_optimize(f, images, fig):
    inital_guss = numpy.zeros((image_size, image_size), dtype=int)

    def callback(iteration_num, state):
        if iteration_num % 10000 == 0:
            save_res(state, images, fig)
    '''
    def no_minimizer(fun, x0, args, jac, hess, hessp,
                      bounds, constraints,
                      callback, *options):
        return optimize.OptimizeResult(x=x0, success=True, fun=fun(x0))

    minimizer_kwargs = {"method": no_minimizer}

    # todo constant temp
    #res = optimize.basinhopping(f, x0=inital_guss, take_step=mytakestep, disp=True, minimizer_kwargs=minimizer_kwargs, niter=10**7, niter_success=1000).x
    '''
    res = simulated_annealing(f, mytakestep, 10**7, 20000, inital_guss, callback)
    return res


def simulated_annealing(f, take_step, niter, niter_success, x0, callback):
    state = x0
    val = f(state)
    global_min_val = val
    global_min_state = x0
    success_iter = 0
    for iteration_num in range(niter):
        if success_iter > niter_success:
            break
        temp = niter - iteration_num - 1
        candidate_state = take_step(state)
        candidate_val = f(candidate_state)

        if candidate_val < val or random.random() <= math.exp((val-candidate_val)/temp):
            state = candidate_state
            val = candidate_val
        print("iteration={0} val={1} global val={2} success_iter={3}".
              format(iteration_num, int(val), int(global_min_val), success_iter))

        if global_min_val > val:
            global_min_val = val
            global_min_state = state
            print("\tfound new global min iteration={0} val={1} prev success_ite={2}".
                  format(iteration_num, int(global_min_val), success_iter))
            success_iter = 0

        else:
            success_iter += 1

        callback(iteration_num, state)
    return global_min_state


if __name__ == "__main__":
    main()
