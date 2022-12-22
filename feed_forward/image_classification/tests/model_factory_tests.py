import pytest

from image_classification import model_factory, InvalidModelFactoryArgument


def test_model_factory_happy_case():
    model_class = "n_layer_nn.N_Layer_NN"
    model = model_factory(model_class)
    assert model


def test_model_factory_no_class_provided():
    model_class = None
    with pytest.raises(InvalidModelFactoryArgument):
        model_factory(model_class)


def test_model_factoy_invalid_module():
    model_class = "feed_forward.image_classification.N_Layer_NN"
    with pytest.raises(InvalidModelFactoryArgument):
        model_factory(model_class)


def test_model_factoy_invalid_class():
    model_class = "n_layer_nn.n_layer_nn"
    with pytest.raises(InvalidModelFactoryArgument):
        model_factory(model_class)
