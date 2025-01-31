import numpy as np
import torch
from aitunes.autoencoders.task_cases import AutoencoderTaskCase, FLAG_NONE

class LinearVectorAugmentationTaskCase(AutoencoderTaskCase):

    @staticmethod
    def expand_3d_vector(val: np.array):
        if not isinstance(val, np.ndarray):
            val = np.array(val)
        return np.column_stack((val[:, 0], (val[:, 0] + val[:, 1] + val[:, 2]) / 3, val[:, 1], -val[:, 2] - val[:, 1] + 2 * val[:, 0], val[:, 2]))
    
    def __init__(self, model, weights_path, loss, optimizer, flags: int = FLAG_NONE):
        super().__init__("5D_Dependency", model, weights_path, loss, optimizer, flags)
        self.training_data = self.expand_3d_vector(np.random.rand(5000, 3))
        self.evaluation_data = self.expand_3d_vector(np.random.rand(200, 3))

    def next_batch(self, training):
        # Yielding 10 bursts of 100 5-d vectors
        dataset = self.training_data
        if not training:
            dataset = self.evaluation_data

        np.random.shuffle(dataset)
        batch_size = 100
        for i in range(dataset.shape[0] // batch_size):
            yield torch.tensor(dataset[i * batch_size:(i + 1) * batch_size, :], dtype=torch.float32), None
        
    def interactive_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            try:
                while True:
                    usr = input("Enter a 3 value vector (e.g. 0.3,0.2,0.1) or 'q' to exit:")
                
                    if usr == 'q':
                        break
                    
                    vector = list(map(lambda x: float(x), usr.split(",")))
                    if len(vector) != 3:
                        continue
                    
                    vector = torch.tensor(self.expand_3d_vector([vector]), dtype=torch.float32)
                    embedding, prediction, *args = self.model(vector)
                    loss = self._loss_criterion(prediction, vector, *args)
                    print(f"Test Case: \n\tInp: {vector.tolist()}\n\tOut: {prediction.tolist()}\n\tLoss: {loss.item()}")
            except:
                self._support.log("An error occurred, exiting interactive mode.")
          