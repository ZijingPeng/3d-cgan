from src.trainer import *
from src.dataset import *

# train
if __name__ == '__main__':
    dataset = Dataset1(prob=0.8)
    stage = Stage1()
    stage.train(dataset, 1000)

    #dataset = Dataset2(prob=0.8)
    #stage = Stage2()
    #stage.train(dataset, 2000)

