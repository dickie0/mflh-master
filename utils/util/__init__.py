import numpy as np

def sign(x):
    y = np.sign(x)
    tmp = y[y == 0]
    y[y==0] = np.random.choice([-1, 1], tmp.shape)
    return y

if __name__ == "__main__":
    x = np.random.choice([-1, 0, 1], [3, 3])
    print(x)
    print(sign(x))
