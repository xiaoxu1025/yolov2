from keras.layers import Conv2D, Add, ZeroPadding2D, Lambda, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import reduce, wraps
from exception import ValueEmptyException
import config as cfg
import tensorflow as tf

# load cfg
cell_size = cfg.CELL_SIZE
classes_num = cfg.CLASSES_NUM
b = cfg.B
iou_threshold = cfg.IOU_THRESHOLD
anchors_num = cfg.ANCHORS_NUM


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    # reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueEmptyException('compose func args empty not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    # LeakyReLU(BatchNormalization(DarknetConv2D))
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(inputs, trainable):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(inputs)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    body_model = Model(inputs=inputs, outputs=x)
    for layer in body_model.layers:
        layer.trainable = trainable
    return body_model


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_model(inputs, trainable=True):
    darknet = darknet_body(inputs, trainable)
    _, y = make_last_layers(darknet.output, 512, anchors_num * (classes_num + 5))
    model = Model(darknet.input, outputs=y)
    return model


# conv2d_59 due to mismatch in shape ((1, 1, 1024, 125) vs (255, 1024, 1, 1)).
# from keras.layers import Input
#
# model = yolo_model(Input(shape=(416, 416, 3)), trainable=True)
# for layer in model.layers:
#     print(layer.trainable)
# model.summary()


# def model_eval(outputs, im_size, iou_threshold=.5):
#     """
#     模型输出
#     :param outputs:
#     :return:
#     """
#     # 边框回归值
#     # (1, 7, 7, 8)
#     bboxes = outputs[..., 0:8]
#     # 置信度
#     # (1, 7, 7, 2)
#     confidences = outputs[..., 8:10]
#     # 类别
#     # (1, 7, 7, 20)
#     classes = outputs[..., 10:]
#     # (1, 7, 7, 2, 1)
#     confidences = tf.expand_dims(confidences, axis=-1)
#     # (1, 7, 7, 1, 20)
#     classes = tf.expand_dims(classes, axis=-2)
#     # (1, 7, 7, 2, 20)
#     scores = confidences * classes
#     # (1, 7, 7, 2, 4)
#     bboxes = tf.reshape(bboxes, shape=(-1, cell_size, cell_size, b, 4))
#
#     # bboxes 回归
#     grid_x = tf.range(cell_size, dtype=tf.float32)
#     grid_y = tf.range(cell_size, dtype=tf.float32)
#     grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
#     x_offset = tf.reshape(grid_x, (-1, 1))
#     y_offset = tf.reshape(grid_y, (-1, 1))
#     x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
#     x_y_offset = tf.cast(tf.reshape(x_y_offset, [1, cell_size, cell_size, 1, 2]), tf.float32)
#
#     # (1, 7, 7, 2, 4)
#     bboxes_normal = tf.stack([
#         (bboxes[..., 0] + x_y_offset[..., 0]) / cell_size * im_size[0],
#         (bboxes[..., 1] + x_y_offset[..., 1]) / cell_size * im_size[1],
#         tf.square(bboxes[..., 2]) * im_size[0],
#         tf.square(bboxes[..., 3]) * im_size[1]
#     ], axis=-1)
#
#     max_boxes_tensor = tf.constant(20, dtype=tf.int32)
#     # (7 * 7 * 2, 4)
#     bboxes_normal = tf.reshape(bboxes_normal, (-1, 4))
#
#     scores = tf.reshape(scores, (-1, 20))
#     # 得到最大值所在的索引
#     argmax_scores = tf.argmax(scores, axis=-1)
#     # (7 * 7 * 2, 1)
#     class_box_scores = tf.gather(scores, argmax_scores, axis=-1)
#     class_box_scores = tf.squeeze(class_box_scores, axis=-1)
#     nms_index = tf.image.non_max_suppression(bboxes_normal, class_box_scores, max_boxes_tensor,
#                                              iou_threshold=iou_threshold)
#     bboxes_normal = tf.gather(bboxes_normal, nms_index)
#     scores = tf.gather(class_box_scores, nms_index)
#     classes = tf.gather(argmax_scores, nms_index)
#     return bboxes_normal, scores, classes

# import numpy as np
#
# a = np.array([[0.1, 0.8]])
# b = np.array([[0.1, 0.1, 0.3, 0.4, 0.1]])
# a = a.reshape((1, 2, 1))
# b = b.reshape((1, 1, 5))
# c = a * b
# print(c, c.shape)

# import tensorflow as tf
#
# a = tf.Variable([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
# index_a = tf.Variable([0])
# #
# # b = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # index_b = tf.Variable([2, 4, 6, 8])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     output = sess.run(tf.gather(a, index_a, axis=-1))
#
#     print(output, output.shape)
# print(sess.run(tf.gather(b, index_b)))
