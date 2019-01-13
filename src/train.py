from src.trainer import Stage
from src.dataset import Dataset

# train
if __name__ == '__main__':
    dataset = Dataset(prob=0.8)
    stage = Stage()
    stage.train(dataset, 100)
