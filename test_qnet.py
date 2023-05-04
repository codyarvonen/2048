import numpy as np
import torch
from qnet import QNet
import unittest


class TestGameFunctions(unittest.TestCase):
    def setUp(self):
        
        self.test_encode_before = np.array([[0, 2, 4, 8], 
                                            [16, 32, 64, 128], 
                                            [256, 512, 1024, 2048], 
                                            [4096, 8192, 16384, 32768]])
        
        self.test_encode_after = torch.tensor([[[[1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 1., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 1., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 1.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 1.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 1.],
                [0., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 0., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 1., 0., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 0.]],
                [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 1.]]]])

    # def test_encode_state(self):
    #     net = QNet(16, 4)
    #     encoded_state = net.encode_state(self.test_encode_before)
    #     print(self.test_encode_after.shape, encoded_state.shape)
    #     self.assertTrue(torch.equal(self.test_encode_after, encoded_state))

    def test_forward(self):
        net = QNet(16, 4)
        net.eval()
        net(self.test_encode_before)



if __name__ == '__main__':
    unittest.main()
