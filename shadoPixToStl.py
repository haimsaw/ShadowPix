from stl import mesh
import math
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
light_angle = math.radians(30)

# todo heits are in s


"""
self.add_to_rectangles_list(
                    [[0, 0, 0],
                    ])
"""


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


class LocalHeightfieldToStl(HeightfieldToStl):
    def __init__(self, vertical_casters_heightfield, horisontal_casters_heightfield, pixels_heightfield, left_chamfers,
                       right_chamfers):
        HeightfieldToStl.__init__(self)
        # casters_heightfield is  a height matrix of size n+1Xm+1
        # pixels_heightfield is a nXm matrix
        # chamfers is a nXm matrix
        self.vertical_casters_heightfield = vertical_casters_heightfield
        self.horisontal_casters_heightfield = horisontal_casters_heightfield
        self.pixels_heightfield = pixels_heightfield
        self.left_chamfers = left_chamfers
        self.right_chamfers = right_chamfers


        self.num_pixels_in_top_dir = len(self.pixels_heightfield)
        self.num_pixels_in_right_dir = len(self.pixels_heightfield[0])

    def generate(self):

        # todo verify that the chamfers directions are correct
        # todo add size asserts
        # todo between casters?


        # region left to right
        distance_from_top = 0
        for i in range(self.num_pixels_in_top_dir):
            distance_from_left = 0

            # region caster row
            for j in range(self.num_pixels_in_right_dir):
                corner_height = self.get_corner_height(i, j)

                neighbor_height = self.horisontal_casters_heightfield[i][j-1] if j != 0 else 0
                # corner  left
                self.add_to_rectangles_list(
                    [[distance_from_left, distance_from_top, neighbor_height],
                     [distance_from_left, distance_from_top, corner_height],
                     [distance_from_left, distance_from_top+casters_width, corner_height],
                     [distance_from_left, distance_from_top+casters_width, neighbor_height],
                     ])

                neighbor_height = self.vertical_casters_heightfield[i-1][j] if i != 0 else 0

                # corner  up
                self.add_to_rectangles_list(
                    [[distance_from_left, distance_from_top, neighbor_height],
                     [distance_from_left, distance_from_top, corner_height],
                     [distance_from_left+casters_width, distance_from_top, corner_height],
                     [distance_from_left+casters_width, distance_from_top, neighbor_height]
                     ])

                # corner top
                self.add_to_rectangles_list(
                    [[distance_from_top, distance_from_left, corner_height],
                     [distance_from_top + casters_width, distance_from_left, corner_height],
                     [distance_from_top + casters_width, distance_from_left + casters_width, corner_height],
                     [distance_from_top, distance_from_left + casters_width, corner_height]])

                # corner down
                self.add_to_rectangles_list(
                    [[distance_from_top+casters_width, distance_from_left, corner_height],
                     [distance_from_top+casters_width, distance_from_left, self.vertical_casters_heightfield[i][j]],
                     [distance_from_top+casters_width, distance_from_left+casters_width, self.vertical_casters_heightfield[i][j]],
                     [distance_from_top+casters_width, distance_from_left+casters_width, corner_height],
                     ])

                distance_from_left += casters_width

                # casters right
                self.add_to_rectangles_list(
                    [[0, 0, 0],
                     ])
                # caster left
                self.add_to_rectangles_list(
                    [[0, 0, 0],
                     ])
                # caster top
                self.add_to_rectangles_list(
                    [[0, 0, 0],
                     ])

                distance_from_left += pixel_size
                # up caster right
                self.add_to_rectangles_list(
                    [[0, 0, 0],
                     ])
            # endregion
            distance_from_top += casters_width

            # region pixel row
            distance_from_left = 0
            for j in range(self.num_pixels_in_right_dir):
                caster_height = self.vertical_casters_heightfield[i][j]
                pixel_height = self.pixels_heightfield[i][j]
                left_chamfer_height = self.left_chamfers[i][j] + pixel_height
                left_chamfer_width = self.left_chamfers[i][j] / mpmath.cot(light_angle)
                right_chamfer_width = self.right_chamfers[i][j] / mpmath.cot(light_angle)
                pixel_top_widt = pixel_size - right_chamfer_width
                left_neighbor_height = self.pixels_heightfield[i][j - 1] + self.right_chamfers[i][j - 1] if j != 0 else 0
                right_chamfer_height = self.right_chamfers[i][j] + pixel_height

                # caster left
                self.add_to_rectangles_list(
                                       [[distance_from_top, distance_from_left, left_neighbor_height],
                                        [distance_from_top + pixel_size, distance_from_left, left_neighbor_height],
                                        [distance_from_top + pixel_size, distance_from_left, caster_height],
                                        [distance_from_top, distance_from_left, caster_height]])
                # caster top
                self.add_to_rectangles_list(
                                       [[distance_from_top, distance_from_left, caster_height],
                                        [distance_from_top + pixel_size, distance_from_left, caster_height],
                                        [distance_from_top + pixel_size, distance_from_left + casters_width,
                                         caster_height],
                                        [distance_from_top, distance_from_left + casters_width, caster_height]])

                distance_from_left += casters_width

                # caster right
                self.add_to_rectangles_list(
                                       [[distance_from_top, distance_from_left, caster_height],
                                        [distance_from_top + pixel_size, distance_from_left, caster_height],
                                        [distance_from_top + pixel_size, distance_from_left, left_chamfer_height],
                                        [distance_from_top, distance_from_left, left_chamfer_height]])
                # caster down


                # left chamfer
                self.add_to_rectangles_list(
                                       [[distance_from_top, distance_from_left, left_chamfer_height],
                                        [distance_from_top + pixel_size, distance_from_left, left_chamfer_height],
                                        [distance_from_top + pixel_size, distance_from_left + left_chamfer_width,
                                         pixel_height],
                                        [distance_from_top, distance_from_left + left_chamfer_width, pixel_height]])

                # pixel height
                self.add_to_rectangles_list(
                                       [[distance_from_top, distance_from_left + left_chamfer_width, pixel_height],
                                        [distance_from_top + pixel_size, distance_from_left + left_chamfer_width,
                                         pixel_height],
                                        [distance_from_top + pixel_size, distance_from_left + pixel_top_widt,
                                         pixel_height],
                                        [distance_from_top, distance_from_left + pixel_top_widt, pixel_height]])

                # right chamfer
                self.add_to_rectangles_list(
                                       [[distance_from_top, distance_from_left + pixel_top_widt, pixel_height],
                                        [distance_from_top + pixel_size, distance_from_left + pixel_top_widt,
                                         pixel_height],
                                        [distance_from_top + pixel_size, distance_from_left + pixel_size,
                                         right_chamfer_height],
                                        [distance_from_top, distance_from_left + pixel_size, right_chamfer_height]])
                # up caster up
                # up caster bottom

                distance_from_left += pixel_size
            # endregion
            distance_from_top += pixel_size
        # endregion

        # region rightmost
        distance_from_top = 0
        distance_from_left = self.num_pixels_in_right_dir * (pixel_size + casters_width)
        j = self.num_pixels_in_right_dir
        for i in range(self.num_pixels_in_top_dir):
            corner_height = self.get_corner_height(i, j)
            caster_height = self.vertical_casters_heightfield[i][j]

            # region rightmost corner
            # corner left
            self.add_to_rectangles_list(
                [[distance_from_top, distance_from_left, self.horisontal_casters_heightfield[i][j]],
                 [distance_from_top+casters_width, distance_from_left, self.horisontal_casters_heightfield[i][j]],
                 [distance_from_top+casters_width, distance_from_left, corner_height],
                 [distance_from_top, distance_from_left, corner_height]])

            # corner up
            neighbor_hight = self.vertical_casters_heightfield[i-1][j] if i > 0 else 0
            self.add_to_rectangles_list(
                [[distance_from_top, distance_from_left, neighbor_hight],
                 [distance_from_top, distance_from_left, corner_height],
                 [distance_from_top, distance_from_left+casters_width, corner_height],
                 [distance_from_top, distance_from_left+casters_width, neighbor_hight],
                 ])

            # corner top
            self.add_to_rectangles_list(
                [[distance_from_top, distance_from_left, corner_height],
                 [distance_from_top+casters_width, distance_from_left, corner_height],
                 [distance_from_top+casters_width, distance_from_left+casters_width, corner_height],
                 [distance_from_top, distance_from_left+casters_width, corner_height]])


            # corner right
            self.add_to_rectangles_list(
                [[distance_from_top, distance_from_left+casters_width, corner_height],
                 [distance_from_top+casters_width, distance_from_left+casters_width, corner_height],
                 [distance_from_top+casters_width, distance_from_left+casters_width, 0],
                 [distance_from_top, distance_from_left+casters_width, 0]])

            # corner down
            self.add_to_rectangles_list(
                [[distance_from_top+casters_width, distance_from_left, corner_height],
                 [distance_from_top+casters_width, distance_from_left, caster_height],
                 [distance_from_top+casters_width, distance_from_left+casters_width, caster_height],
                 [distance_from_top+casters_width, distance_from_left+casters_width, corner_height],
                 ])
            # endregion
            distance_from_top += casters_width

            # region rightmost caster
            # rightmost caster right
            self.add_to_rectangles_list(
                [[distance_from_top, distance_from_left+casters_width, 0],
                 [distance_from_top, distance_from_left+casters_width, caster_height],
                 [distance_from_top+pixel_size, distance_from_left+casters_width, caster_height],
                 [distance_from_top+pixel_size, distance_from_left+casters_width, 0]
                 ])

            # rightmost caster top
            self.add_to_rectangles_list(
                [[distance_from_top, distance_from_left, caster_height],
                 [distance_from_top+pixel_size, distance_from_left, caster_height],
                 [distance_from_top+pixel_size, distance_from_left+casters_width, caster_height],
                 [distance_from_top, distance_from_left+casters_width, caster_height],
                 ])

            # rightmost caster left
            # endregion
            distance_from_top += pixel_size
        # endregion

        # region downmost
        for j in range(self.num_pixels_in_right_dir):
            pass
            # downmost corner
            # downmost caster
        # endregion

        # region downmost righmost corner
        # downmost righmost corner
        # endregion
        '''
        # region bottom
        grid_width = self.num_pixels_in_right_dir * (pixel_size + casters_width) + casters_width
        grid_height = self.num_pixels_in_top_dir * (pixel_size + casters_width) + casters_width
        self.add_to_rectangles_list([[0, 0, 0],
                                    [0, grid_width, 0],
                                    [grid_height, grid_width, 0],
                                    [grid_height, 0, 0]])
        # endregion
        '''

        my_mash = self.rectangles_to_mash()
        return my_mash

    def get_corner_height(self, i, j):
        neighbors = [self.horisontal_casters_heightfield[i][j], self.vertical_casters_heightfield[i][j]]

        if i > 0:
            neighbors.append(self.horisontal_casters_heightfield[i-1][j])
        elif i < self.num_pixels_in_top_dir -1:
            neighbors.append(self.horisontal_casters_heightfield[i+1][j])

        if j > 0:
            neighbors.append(self.vertical_casters_heightfield[i][j-1])
        elif i < self.num_pixels_in_right_dir - 1:
            neighbors.append(self.vertical_casters_heightfield[i][j+1])

        return sum(neighbors)/len(neighbors)


class GlobalHeightfieldToStl(HeightfieldToStl):
    def __init__(self, heightfield):
        # heightfield is just a height matrix of size nXm
        HeightfieldToStl.__init__(self)
        self.heightfield = heightfield

    def generate(self):

        # todo add edge

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
        self.add_to_rectangles_list( [[0, 0, 0],
                                        [0, grid_width, 0],
                                        [grid_height, grid_width, 0],
                                        [grid_height, 0, 0]])

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
    my_mash = GlobalHeightfieldToStl(heightfield).generate()
    my_mash.save('test.stl')
    render(my_mash)

def test_local1():
    vertical_casters_heightfield = [[2, 2]]
    horisontal_casters_heightfield = [[3, 3]]
    pixels_heightfield = [[1]]
    left_chamfers = [[0.5]]
    right_chamfers = [[0.75]]

    my_mash = LocalHeightfieldToStl(vertical_casters_heightfield, horisontal_casters_heightfield,
                             pixels_heightfield, left_chamfers, right_chamfers).generate()
    my_mash.save('test.stl')
    render(my_mash)


def test_local2():
    vertical_casters_heightfield = [[4, 3, 2],
                                    [2, 5, 3],
                                    [4, 3, 2]]
    left_chamfers = [[0.5, 0.3],
                     [0.1, 0.3]]

    horisontal_casters_heightfield = [[2, 2, 1],
                                      [2, 2, 3],
                                      [2, 2, 1]]
    pixels_heightfield = [[2, 1],
                          [1, 2]]

    right_chamfers = [[0.5, 0.3],
                      [0.1, 0.3]]

    my_mash = LocalHeightfieldToStl(vertical_casters_heightfield, horisontal_casters_heightfield,
                             pixels_heightfield, left_chamfers, right_chamfers).generate()
    my_mash.save('test.stl')
    render(my_mash)


if __name__ == "__main__":
    test_global()
    # test_local2()


