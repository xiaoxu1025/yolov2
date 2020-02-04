import os
import cv2
import numpy as np
import colorsys
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from exception import ResourceException, ValueEmptyException
from keras.utils import multi_gpu_model
from keras import backend as K
import config as cfg
from model.model import yolo_model
from keras.layers import Input
from model.model import model_eval

im_size = cfg.IM_SIZE
img_channel_mean = cfg.IMG_CHANNEL_MEAN
img_scaling_factor = cfg.IMG_SCALING_FACTOR
classes_num = cfg.CLASSES_NUM


class YOLO(object):

    def __init__(self, model_path, cls_path='./data/voc_classes.txt', gpu=False, gpu_num=0):
        self._model_path = model_path
        self._gpu = gpu
        self._gpu_num = gpu_num
        self._sess = K.get_session()
        self._class_names = self._get_class_names(cls_path)
        self._input_tensor = Input(shape=(None, None, 3))
        self._input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = self._generate()

    def _generate(self):
        model = yolo_model(self._input_tensor)
        if not os.path.exists(self._model_path):
            raise ValueEmptyException('model path not exist')
        model.load_weights(self._model_path)

        print('{} model, anchors, and classes loaded.'.format(self._model_path))

        hsv_tuples = [(x / len(self._class_names), 1., 1.)
                      for x in range(len(self._class_names))]
        self._colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self._colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self._colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self._colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        if self._gpu and self._gpu_num > 0:
            model = multi_gpu_model(model, gpus=self._gpu_num)
        self._model = model
        # model.predict
        boxes, scores, classes = model_eval(self._model.output, self._input_image_shape)
        return boxes, scores, classes

    def _get_class_names(self, cls_path):
        class_names = open(cls_path).readlines()
        class_names = [class_name.strip() for class_name in class_names]
        return class_names

    def detect_image(self, image):
        start = timer()
        # 原图大小
        width, height = image.size
        image = image.resize(im_size)
        image_data = np.array(image, dtype='float32')
        image_data[:, :, 0] -= img_channel_mean[0]
        image_data[:, :, 1] -= img_channel_mean[1]
        image_data[:, :, 2] -= img_channel_mean[2]
        image_data /= img_scaling_factor
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        out_boxes, out_scores, out_classes = self._sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self._input_tensor: image_data,
                self._input_image_shape: [width, height],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self._class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self._colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self._colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self._sess.close()

    def detect_video(self, video_path, output_path=''):
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise ResourceException('video {} could not open!'.format(video_path))
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != '' else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = self.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.close_session()
