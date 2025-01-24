import torch.optim as optim
import torch.nn as nn


from autoencoders_modules import SimpleAutoEncoder
from test_cases import *
    

def example1(model_weights: str = "02_AutoEncoders/models/ae_5d_vectors.pth"):
    model = SimpleAutoEncoder((5, 4, 3))
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_PLOTTING
    test_case = LinearVectorAugmentationTestCase(model, model_weights, loss, optimizer, flags)
    
    if not test_case.trained:
        test_case.train(200)
    test_case.evaluate()
    test_case.interactive_evaluation()


def example2(model_weights: str = "02_AutoEncoders/models/ae_mnist.pth"):
    model = SimpleAutoEncoder((28 * 28, 14 * 14, 7 * 7, 3 * 3, 2))
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    flags = FLAG_PLOTTING
    test_case = MnistDigitCompressionTestCase(model, model_weights, loss, optimizer, flags)
    
    if not test_case.trained:
        test_case.train(10)
    test_case.evaluate()
    test_case.display_embed_plot()
    test_case.interactive_evaluation()


def main(): 
    # example1()
    example2()

if __name__ == "__main__":
    main()
