import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE


class make_dataset:
    def __init__(self,
                 lr_img_paths,
                 hr_img_paths,
                 scale=4,
                 downgrade='bicubic'):

        self.lr_img_paths = lr_img_paths
        self.hr_img_paths = hr_img_paths
        self.scale = scale
        self.downgrade = downgrade

    def dataset(self, batch_size, repeat_count=None, random_transform=False):
        lr_dataset = self._images(self.lr_img_paths)
        hr_dataset = self._images(self.hr_img_paths)

        dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

        if random_transform:
            dataset = dataset.map(lambda lr, hr: random_crop(lr, hr, scale=self.scale), num_parallel_calls=AUTOTUNE)
            dataset = dataset.map(random_rotate, num_parallel_calls=AUTOTUNE)
            dataset = dataset.map(random_flip, num_parallel_calls=AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset

    @staticmethod
    def _images(img_paths):
        imgs = tf.data.Dataset.from_tensor_slices(img_paths)
        imgs = imgs.map(tf.io.read_file)
        imgs = imgs.map(lambda x: tf.image.decode_png(x, 3), num_parallel_calls=AUTOTUNE)
        return imgs


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=4):
    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)