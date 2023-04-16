from net import CustomNeuralNetwork
import data_loader

nn = CustomNeuralNetwork([784, 30, 10])
nn.stochastic_gradient_descent(30, data_loader.training_loader,
                               data_loader.testing_loader, len(data_loader.testing_data), 3.0)



# import mnist_loader
# import net2
# training_data, test_data = mnist_loader.load_data_wrapper()
# net = net2.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)