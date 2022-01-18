import inspect


def get_model_inputs(model_function, inputs):
    if isinstance(model_function, list):
        model_inputs = model_function
    else:
        model_inputs = inspect.signature(model_function).parameters.keys()
    return {model_input: inputs[model_input] for model_input in model_inputs}
