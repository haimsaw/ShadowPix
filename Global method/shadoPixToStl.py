from stl import mesh
from math import  radians
import numpy
import random
import mpmath
from matplotlib import pyplot
from mpl_toolkits import mplot3d

# (0,0) it top left, i is down, j is right

# params
pixel_size = 1

# local method params
casters_width = 0.5
light_angle = radians(30)


class HeightfieldToStl:
    def __init__(self):
        self.rectangles = []

    def rectangles_to_mash(self):
        data = numpy.zeros(len(self.rectangles) * 2, dtype=mesh.Mesh.dtype)
        for i in range(len(self.rectangles)):
            self.add_to_mash_data(i, data)
        return mesh.Mesh(data.copy())

    def add_to_rectangles_list(self, req):
        # don't add rectangles with area 0
        # assuming all rectangles have straight angles
        # todo check rew is not here
        if len(req) < 4:
            return
        if req[1] != req[0] and req[1] != req[2]:
            self.rectangles.append(req)

    def add_to_mash_data(self, index, mash_data):
        triangles = self.rectangle_to_triangles(self.rectangles[index][0], self.rectangles[index][1], self.rectangles[index][2], self.rectangles[index][3])
        mash_data['vectors'][index] = triangles[0]
        mash_data['vectors'][index + len(self.rectangles)] = triangles[1]

    def rectangle_to_triangles(self, p0, p1, p2, p3):
        # edges are from p0->p1, p1 -> p2, p2->p3, p3->p0
        # splitting at p0-p2
        return numpy.array([p0, p1, p2]), numpy.array([p0, p2, p3])


class GlobalHeightfieldToStl(HeightfieldToStl):
    def __init__(self, heightfield):
        # heightfield is just a height matrix of size nXm
        HeightfieldToStl.__init__(self)
        self.heightfield = heightfield

    def generate(self):

        grid_height = len(self.heightfield)
        grid_width = len(self.heightfield[0])

        for i in range(grid_height):
            for j in range(grid_width):
                height_at_pixel = self.heightfield[i][j]

                left_neighbor_height = self.heightfield[i][j - 1] if j != 0 else 0
                up_neighbor_height = self.heightfield[i - 1][j] if i != 0 else 0

                # left
                self.add_to_rectangles_list( [[i, j, left_neighbor_height],
                                                         [i + 1, j, left_neighbor_height],
                                                         [i + 1, j, height_at_pixel],
                                                         [i, j, height_at_pixel]])

                # up
                self.add_to_rectangles_list( [[i, j, height_at_pixel],
                                                [i, j + 1, height_at_pixel],
                                                [i, j + 1, up_neighbor_height],
                                                [i, j, up_neighbor_height]])

                # top
                self.add_to_rectangles_list( [[i, j, height_at_pixel],
                                                [i + 1, j, height_at_pixel],
                                                [i + 1, j + 1, height_at_pixel],
                                                [i, j + 1, height_at_pixel]])

        # rightmost
        for i in range(grid_height):
            height_at_pixel = self.heightfield[i][grid_width - 1]
            self.add_to_rectangles_list( [[i, grid_width, 0],
                                            [i, grid_width, height_at_pixel],
                                            [i + 1, grid_width, height_at_pixel],
                                            [i + 1, grid_width, 0]])

        # downmost
        for j in range(grid_width):
            height_at_pixel = self.heightfield[grid_height - 1][j]
            self.add_to_rectangles_list( [[grid_height, j, 0],
                                            [grid_height, j + 1, 0],
                                            [grid_height, j + 1, height_at_pixel],
                                            [grid_height, j, height_at_pixel]])
        # bottom
        self.add_to_rectangles_list( [[0, 0, -1],
                                        [0, grid_width, -1],
                                        [grid_height, grid_width, -1],
                                        [grid_height, 0, -1]])

        self.add_to_rectangles_list( [[0, 0, 0],
                                        [0, 0, -1],
                                        [0, grid_width, -1],
                                        [0, grid_width, 0]])

        self.add_to_rectangles_list( [[0, 0, 0],
                                        [0, 0, -1],
                                        [grid_height, 0, -1],
                                        [grid_height, 0, 0]])

        self.add_to_rectangles_list( [[grid_height, 0, 0],
                                        [grid_height, 0, -1],
                                        [grid_height, grid_width, -1],
                                        [grid_height, grid_width, 0]])

        self.add_to_rectangles_list([[0, grid_width, 0],
                                     [0, grid_width, -1],
                                     [grid_height, grid_width, -1],
                                     [grid_height, grid_width, 0]])

        # todo for printer
        self.add_to_rectangles_list([[140, 140, -50],
                                     [140, 160, -50],
                                     [160, 160, -50],
                                     [160, 140, -50]])
        # todo for printer

        self.add_to_rectangles_list([[140, 140, 0],
                                     [140, 140, -50],
                                     [140, 160, -50],
                                     [140, 160, 0]])
        # todo for printer

        self.add_to_rectangles_list([[140, 140, 0],
                                     [140, 140, -50],
                                     [160, 140, -50],
                                     [160, 140, 0]])
        # todo for printer

        self.add_to_rectangles_list([[160, 140, 0],
                                     [160, 140, -50],
                                     [160, 160, -50],
                                     [160, 160, 0]])
        # todo for printer

        self.add_to_rectangles_list([[140, 160, 0],
                                     [140, 160, -50],
                                     [160, 160, -50],
                                     [160, 160, 0]])

        my_mash = self.rectangles_to_mash()

        # adjust for pixel size
        my_mash.x *= pixel_size
        my_mash.y *= pixel_size

        return my_mash


def render(my_mash):
    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(my_mash.vectors))

    # Auto scale to the mesh size
    scale = my_mash.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()


def test_global():
    heightfield_global1 = [[1, 2],
                           [3, 4]]
    heightfield_global2 = [[2, 1],
                           [3, 4]]
    heightfield_global3 = [[random.uniform(0, 5) for _ in range(5)] for _ in range(6)]
    my_mash = GlobalHeightfieldToStl(heightfield_global3).generate()
    my_mash.save('test.stl')
    render(my_mash)


def create_stl_global(heightfield):
    print("converting heightfield to stl printable file")
    s=mpmath.cot(light_angle)
    heightfield = heightfield*s
    my_mash = GlobalHeightfieldToStl(heightfield).generate()
    my_mash.save('res.stl')
    # render(my_mash)


if __name__ == "__main__":
    heightfield = numpy.load("res_heightfield.npy")
    create_stl_global(heightfield)
    #test_global()
    # test_local2()


