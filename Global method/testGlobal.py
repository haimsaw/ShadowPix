from GenareteGlobal import *
import numpy


class TestIsLit:
    @staticmethod
    def test1():
        heightfield = numpy.array(
                        [[1, 0],
                       [0, 0]])
        direction_vector = Direction("-x").get_direction_vector()
        assert is_lit(0, 1, direction_vector, heightfield) == True

    @staticmethod
    def test2():
        heightfield = numpy.array(
                        [[0, 1],
                       [0, 0]])
        direction_vector = Direction("-x").get_direction_vector()
        assert is_lit(0, 1, direction_vector, heightfield) == False


TestIsLit.test1()
TestIsLit.test2()