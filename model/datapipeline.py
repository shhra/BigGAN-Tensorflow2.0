import pathlib
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


AUTOTUNE = tf.data.experimental.AUTOTUNE


class InputHandler:
    """
    Creates tf.records from the images at given location
    """
    def __init__(self, root_path):
        """
        Initialize input handler
        :param root_path: Get the root location of dataset
        """
        random.seed(123112)
        self.data_path = pathlib.Path(root_path)
        self.path_list = self.shuffled_image_path()
        self.indexed_labels = self.generate_labels_with_index()

    def shuffled_image_path(self):
        """
        Iterates through the root directory, adds images into list
        :return: list of shuffled image path
        """
        image_path_list = list(self.data_path.glob('*/*'))
        image_path_list = [str(path) for path in image_path_list if path.name.endswith('.jpg')]
        random.shuffle(image_path_list)
        return image_path_list

    def generate_labels_with_index(self):
        """
        Uses the sub-directories inside the root_path to generate images. The images
        are assigned the corresponding indexed label.

        :return: a list of indexed labels in alphabetical order
        """
        labels = sorted(item.name for item in self.data_path.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(labels))
        image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in
                        self.path_list]
        return image_labels

    @staticmethod
    def preprocess_image(image):
        """
        Takes an image tensor and preprocess it
        :param image: A Tensorflow image
        :return: preprocessed image
        """
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [256, 256])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image /= 255.0
        return image

    def load_and_preprocess_image(self, path):
        """
        Reads image from the path and return a preprocessed image
        :param path: Path where input is located
        :return: preprocessed image
        """
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def build_dataset(self):
        # TODO: create a dataset generator
        pass

    @staticmethod
    def parse(x):
        """
        De-serialize tensors within tf.records
        :return: deserialized tensors
        """
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        result = tf.reshape(result, [256, 256, 3])
        return result

    def build_tf_record(self):
        """
        Creates a serialized tf records from the image paths
        """
        path_ds = tf.data.Dataset.from_tensor_slices(self.path_list)
        image_ds = path_ds.map(self.load_and_preprocess_image)
        ds = image_ds.map(tf.io.serialize_tensor)
        print(ds)
        tfrec = tf.data.experimental.TFRecordWriter('data_256x256.tfrec')
        tfrec.write(ds)

    def build_ds_from_record(self):
        """
        Uses tf.records to build the dataset to be fed into the network
        :return:
        """
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.indexed_labels, tf.int64))
        ds = tf.data.TFRecordDataset('data_256x256.tfrec')
        ds = ds.map(self.parse, num_parallel_calls=AUTOTUNE)
        ds = tf.data.Dataset.zip((ds, label_ds))
        return ds

    def build_ds(self):
        path_ds = tf.data.Dataset.from_tensor_slices(self.path_list)
        image_ds = path_ds.map(self.load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.indexed_labels, tf.int64))
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        image_count = len(self.path_list)
        ds = ds.shuffle(buffer_size=image_count)
        return ds


def plot_image(img, label):
    """
    Plots given image and label
    :param img: A single image
    :param label: Label for the image
    """
    plt.imshow(img)
    plt.grid(False)
    plt.xlabel(label)
    plt.show()


def clean_images(root_path):
    """
    This is essential for cleaning dataset. This function removes corrupted images, or any images
    with format other than jpg.
    :param root_path: Root to dataset
    """
    print("Cleaning")
    path_list = [path for path in pathlib.Path(root_path).glob('*/*') if path.name.endswith('.jpg')]
    for each in path_list:
        try:
            img = Image.open(each)
            exif_data = img._getexif()
            img.verify()
        except (OSError, UserWarning, IOError, ValueError, AttributeError) as e:
            print(each.name)
            each.unlink()


if __name__ == '__main__':
    # clean_images('../../Data/dog_cat')
    input_pipeline = InputHandler(root_path='../../Data/dog_cat')
    ds = input_pipeline.build_ds()
    ds = ds.batch(32).prefetch(AUTOTUNE)
    print(ds)



