import numpy as np


class ConvolutionalNetwork:

    def __init__(self, configure, filter_size=3, d1=16, d2=32, h1=64):

        # get configure.
        self.TOTAL_EPOCH = configure['total_epoch']
        self.BATCH_SIZE = configure['batch_size']
        self.LEARNING_RATE = configure['learning_rate']
        self.TRAIN_DATASET_SIZE = configure['train_dataset_size']
        self.TEST_DATASET_SIZE = configure['test_dataset_size']

        # outdim, indim, h, w
        self.w1 = np.random.randn(filter_size, filter_size, 3, d1) / np.sqrt(3)
        self.b1 = np.random.randn(d1) / np.sqrt(3)

        self.w2 = np.random.randn(filter_size, filter_size, d1, d2) / np.sqrt(d1)
        self.b2 = np.random.randn(d2) / np.sqrt(d1)

        # 8 means 32 - > 16 (pooling) , 16 -> 8 (pooling)
        self.w3 = np.random.randn(8 * 8 * d2, h1) / np.sqrt(d2)
        self.b3 = np.random.randn(h1) / np.sqrt(d2)

        self.w4 = np.random.randn(h1, 10) / np.sqrt(h1)
        self.b4 = np.random.randn(10) / np.sqrt(h1)

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

    def Im2Col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, H, W, C = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                # NOTICE : transpose for calculation.
                col[:, :, y, x, :, :] = np.transpose(img[:, y:y_max:stride, x:x_max:stride, :], [0, 3, 1, 2])

        col = np.reshape(col.transpose([0, 4, 5, 1, 2, 3]), [N * out_h * out_w, -1])
        return col

    def Col2Im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, H, W, C = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        col = np.reshape(col, newshape=[N, out_h, out_w, C, filter_h, filter_w]).transpose([0, 3, 4, 5, 1, 2])

        img = np.zeros((N, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1, C))

        # NOTICE : transpose for calculation.
        img = np.transpose(img, [0, 3, 1, 2])

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        # NOTICE : transpose for calculation.
        img = np.transpose(img, [0, 2, 3, 1])

        return img[:, pad:H + pad, pad:W + pad, :]

    def conv(self, x, weight, bias, stride=1, padding=0):
        filter_h, filter_w, filter_c, filter_d = weight.shape
        N, H, W, C = x.shape

        out_height = int((H + 2 * padding - filter_h) / stride + 1)
        out_width = int((W + 2 * padding - filter_w) / stride + 1)

        flatten_img = self.Im2Col(x, filter_h, filter_w, stride, padding)
        flatten_weight = weight.reshape(filter_d, -1).T
        out = np.dot(flatten_img, flatten_weight) + bias
        out = np.reshape(out, [N, out_height, out_width, -1])

        return out, flatten_img

    def max_pool(self, x, pooling_size=2, stride=2, padding=0):
        N, H, W, C = x.shape
        out_height = int((H - pooling_size) / stride + 1)
        out_width = int((W - pooling_size) / stride + 1)

        flatten_img = self.Im2Col(x, pooling_size, pooling_size, stride, padding)
        # reshape for max.
        flatten_img = flatten_img.reshape(-1, pooling_size * pooling_size)

        # find max val.
        out = np.max(flatten_img, axis=-1)
        # save the max val location.
        loc = np.argmax(flatten_img, axis=-1)
        out = np.reshape(out, [N, out_height, out_width, C])

        return out, loc

    def flat_img(self, x):
        return x.reshape(x.shape[0], -1)

    def FeedForward(self, x):

        y1, flatten_y1 = self.conv(x, self.w1, self.b1, stride=1, padding=1)
        activated_y1 = self.relu(y1)
        pooling_y1, max_loc_y1 = self.max_pool(activated_y1)

        y2, flatten_y2 = self.conv(pooling_y1, self.w2, self.b2, stride=1, padding=1)
        activated_y2 = self.relu(y2)
        pooling_y2, max_loc_y2 = self.max_pool(activated_y2)

        # flat image
        y3 = np.dot(self.flat_img(pooling_y2), self.w3) + self.b3
        activated_y3 = self.relu(y3)

        y4 = np.dot(activated_y3, self.w4) + self.b4
        result = self.softmax(y4)

        return activated_y1, activated_y2, activated_y3, result, pooling_y1, pooling_y2, max_loc_y1, max_loc_y2, flatten_y1, flatten_y2

    def backpropagation(self, x, labelY, activated_y1, activated_y2, activated_y3, result, pooling_y1, pooling_y2,
                        max_loc_y1, max_loc_y2, flatten_y1, flatten_y2):

        # POINT : soft max layer.
        #
        # NOTICE : softmax back.
        error_back = (result - labelY) / len(x)  # divided by batch size.

        # POINT : fully connected layer.
        #
        # NOTICE : fully weight 4 back.
        d_w4 = np.dot(activated_y3.T, error_back)
        d_b4 = np.sum(error_back, axis=0)
        error_back = np.dot(error_back, self.w4.T)

        # NOTICE : ReLU back.
        error_back = self.back_relu(activated_y3) * error_back
        # NOTICE : fully weight 3 back.
        # output x is 4d array, so change the shape.
        d_w3 = np.dot(pooling_y2.reshape(x.shape[0], -1).T, error_back)
        d_b3 = np.sum(error_back, axis=0)
        error_back = np.dot(error_back, self.w3.T)

        # POINT : convolution layer.
        #
        # NOTICE : pooling back.
        # un flat.
        error_back = error_back.reshape([-1, 8, 8, 32])
        # flated pooling
        flated_pool = np.zeros((error_back.size, 4))
        flated_pool[np.arange(len(max_loc_y2)), max_loc_y2.flatten()] = error_back.flatten()
        flated_pool = flated_pool.reshape(error_back.shape + (4,))
        flated_pool = flated_pool.reshape(flated_pool.shape[0] * flated_pool.shape[1] * flated_pool.shape[2], -1)
        error_back = self.Col2Im(flated_pool, [len(x), 16, 16, 32], 2, 2, stride=2, pad=0)

        # NOTICE: Relu back.
        error_back = self.back_relu(activated_y2) * error_back
        # NOTICE : convolution 2 back.
        error_back = error_back.reshape([-1, self.w2.shape[-1]])
        d_w2 = np.dot(flatten_y2.T, error_back)
        d_w2 = np.transpose(d_w2.transpose([1, 0]).reshape([32, 16, 3, 3]), [2, 3, 1, 0])
        d_b2 = np.sum(error_back, axis=0)
        flat_conv = np.dot(error_back, (self.w2.reshape(32, -1).T).T)
        error_back = self.Col2Im(flat_conv, [len(x), 16, 16, 16], 3, 3, stride=1, pad=1)

        # NOTICE : pooling back.
        # un flat.
        error_back = error_back.reshape([-1, 16, 16, 16])
        # flated pooling
        flated_pool = np.zeros((error_back.size, 4))
        flated_pool[np.arange(len(max_loc_y1)), max_loc_y1.flatten()] = error_back.flatten()
        flated_pool = flated_pool.reshape(error_back.shape + (4,))
        flated_pool = flated_pool.reshape(flated_pool.shape[0] * flated_pool.shape[1] * flated_pool.shape[2], -1)
        error_back = self.Col2Im(flated_pool, [len(x), 32, 32, 16], 2, 2, stride=2, pad=0)
        # channel is not changed.

        # NOTICE: Relu back.
        error_back = self.back_relu(activated_y1) * error_back
        # NOTICE : convolution 1 back.
        error_back = error_back.reshape([-1, self.w1.shape[-1]])
        d_w1 = np.dot(flatten_y1.T, error_back)
        d_w1 = np.transpose(d_w1.transpose([1, 0]).reshape([16, 3, 3, 3]), [2, 3, 1, 0])
        d_b1 = np.sum(error_back, axis=0)

        return d_w1, d_w2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4

    def update_weight(self, d_w1, d_w2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4):
        self.w1 -= self.LEARNING_RATE * d_w1
        self.w2 -= self.LEARNING_RATE * d_w2
        self.w3 -= self.LEARNING_RATE * d_w3
        self.w4 -= self.LEARNING_RATE * d_w4
        self.b1 -= self.LEARNING_RATE * d_b1
        self.b2 -= self.LEARNING_RATE * d_b2
        self.b3 -= self.LEARNING_RATE * d_b3
        self.b4 -= self.LEARNING_RATE * d_b4

    # train model.
    def train(self, x, y):
        activated_y1, activated_y2, activated_y3, result, pooling_y1, pooling_y2, max_loc_y1, max_loc_y2, flatten_y1, flatten_y2 = self.FeedForward(
            x)
        d_w1, d_w2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4 = self.backpropagation(x, y, activated_y1, activated_y2,
                                                                              activated_y3, result, pooling_y1,
                                                                              pooling_y2, max_loc_y1,
                                                                              max_loc_y2, flatten_y1, flatten_y2)
        self.update_weight(d_w1, d_w2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4)
        pass

    def predict(self, input):
        output = self.FeedForward(input)
        return output[3]

    def getAccuracyAndLoss(self, output_of_model, output):
        accuracy = np.mean(np.equal(np.argmax(output_of_model, axis=1), np.argmax(output, axis=1)))

        # cross entropy loss
        loss = -np.mean(output * np.log(output_of_model + 1e-7) + (1 - output) * np.log((1 - output_of_model) + 1e-7))

        return accuracy, loss
