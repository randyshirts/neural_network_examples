import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_data(
    train_file: str = "feed_forward/image_classification/datasets/train_catvnoncat.h5",
    test_file: str = "feed_forward/image_classification/datasets/test_catvnoncat.h5",
):
    train_dataset = h5py.File(train_file, "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File(test_file, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def display_dataset_info(training_data=None, training_labels=None, test_labels=None):
    # Explore your dataset
    m_train = training_data.shape[0]
    num_px = training_data.shape[1]
    m_test = training_data.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(training_data.shape))
    print("train_y shape: " + str(training_labels.shape))
    print("test_x_orig shape: " + str(training_data.shape))
    print("test_y shape: " + str(test_labels.shape))


def show_example_image(
    index: int = 15, training_data=None, training_labels=None, classifications=None
) -> None:
    # Show an example picture
    # Example of a picture
    plt.imshow(training_data[index])
    plt.show()

    print(
        "y = "
        + str(training_labels[0, index])
        + ". It's a "
        + classifications[training_labels[0, index]].decode("utf-8")
        + " picture."
    )


def flatten_image_data(original_images: list[int, int, int] = None) -> list:
    # Reshape the training and test examples
    flattened_data = original_images.reshape(
        original_images.shape[0], -1
    ).T  # The "-1" makes reshape flatten the remaining dimensions

    # Standardize data to have feature values between 0 and 1.
    flattened_image_features = flattened_data / 255.0

    return flattened_image_features
