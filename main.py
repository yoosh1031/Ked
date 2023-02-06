from item2vec import Item2Vec
if __name__ == "__main__":
    FILE_PATH = './'
    I2V = Item2Vec(FILE_PATH)
    I2V.run(size=128,class_name="세세분류",topk=5)