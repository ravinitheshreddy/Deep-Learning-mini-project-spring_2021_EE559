# Baseclass for modules


class Module(object):

    """
    Base class for all modules.
    """

    def forward(self, input_):
        """
        Function to get the input, apply forward pass of module and
        returns a tensor or a tuple of tensors.
        """
        raise NotImplementedError

    def backward(self, gradswrtoutput):
        """
        Function to get the input gradient of the loss with respect to the
        module’s output, accumulate the gradient wrt the parameters, and
        return a tensor or a tuple of tensors containing the gradient of
        the loss wrt the module’s input.
        """
        raise NotImplementedError

    def param(self):
        """
        Returns a list of pairs, each composed of a parameter tensor, and
        a gradient tensor of same size.
        """
        return []

    def zero_grad(self):
        """
        Sets the gradients of a module to 0
        """
        return []


# Baseclass for Optimizer
class Optimizer(object):

    """
    Base class for optimizers.
    """

    def step(self):
        """
         Perform the single optimization step
        """

        raise NotImplementedError
