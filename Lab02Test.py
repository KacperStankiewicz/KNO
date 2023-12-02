import unittest
import numpy as np
import Lab02


class TestLab02(unittest.TestCase):
    def test_rotation(self):
        wynik= Lab02.rotate(2,2,90)
        oczekiwane = np.array([[-2],[2]])
        self.assertTrue(np.array_equal(wynik, oczekiwane))


if __name__ == '__main__':
    unittest.main()