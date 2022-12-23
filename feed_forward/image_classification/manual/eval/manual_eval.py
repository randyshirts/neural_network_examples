import matplotlib.pyplot as plt
import numpy as np


def sample_external_image(
    num_pixels: int, parameters: list, classifications: list, model: object
) -> None:
    import imageio.v2 as imageio
    import PIL

    my_image = "IMG_2296.jpg"  # change this to the name of your image file
    my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)

    fname = "feed_forward/image_classification/images/" + my_image
    image = np.array(imageio.imread(fname))
    my_image = np.array(
        PIL.Image.fromarray(image).resize((num_pixels, num_pixels))
    ).reshape((num_pixels * num_pixels * 3, 1))
    my_predicted_image = model.predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    print(
        "y = "
        + str(np.squeeze(my_predicted_image))
        + ', your L-layer model predicts a "'
        + classifications[
            int(np.squeeze(my_predicted_image)),
        ].decode("utf-8")
        + '" picture.'
    )
    plt.show()


def predict(X, y, parameters, forward_func):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    _ = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, _ = forward_func(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p
