import numpy as np


class ConvolutionalNetwork:

    def __init__(self):
        # outdim, indim, h, w
        self.conv1 = np.random.randn(3, 3, 3, 16) / np.sqrt(3)
        self.conv2 = np.random.randn(3, 3, 16, 32) / np.sqrt(16)
        self.w3 = np.random.randn(8 * 8 * 32, 64) / np.sqrt(32)
        self.w4 = np.random.randn(64, 10) / np.sqrt(64)

    def relu(self, x):
        array = np.copy(x)
        array[array < 0] = 0
        return array

    def back_relu(self, x):
        array = np.copy(x)
        array[array >= 0] = 1
        array[array < 0] = 0
        return array

    def softmax(self, x):
        array = np.copy(x)
        if array.ndim == 1:
            array = array.reshape([1, array.size])
        exps = np.exp(array)
        return exps / np.sum(exps, axis=1).reshape([exps.shape[0], 1])

    def padding(self, x):
        # x.shape = (N, H ,W ,C)
        return np.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], 'constant')

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

        Parameters
        ----------
        input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩

        Returns
        -------
        col : 2차원 배열
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose([0, 4, 5, 1, 2, 3]).reshape(N * out_h * out_w, -1)
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

        Parameters
        ----------
        col : 2차원 배열(입력 데이터)
        input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩

        Returns
        -------
        img : 변환된 이미지들
        """
        N, C, H, W = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose([0, 3, 4, 5, 1, 2])

        img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]


    def conv(self):

