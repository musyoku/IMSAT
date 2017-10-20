## Learning Discrete Representations via Information Maximizing Self Augmented Training

- Code for the [paper](https://arxiv.org/abs/1702.08720)
- [実装について](http://musyoku.github.io/2017/03/11/Learning-Discrete-Representations-via-Information-Maximizing-Self-Augmented-Training/)

### Requirements

- Chainer 2

## MNIST

### Unsupervised

```
python train.py
```

#### Results

train data (60,000)

accuracy: 98.4%

```
5889   18    0    2    0    1    3    3    7    0
   1    3   13   27   14 6538    0  145    0    1
   9    2   21    2    1    4    2 5902    5   10
   1    0   16    0   10    2   39   18   24 6021
   6   12    4 5764   48    2    0    2    4    0
   5   30    2    0   22    0 5336    4    7   15
  12 5863    0    2    0    3   29    2    6    1
   4    0 6136   18   48   16    1   38    2    2
   3    9    2   10   13   17   34    7 5748    8
   9    3   20   40 5746    3   15    2   81   30
```

test data (10,000)

accuracy: 98.4%

```
 979    0    1    0    0    0    0    0    0    0
   0    1    1    1    0 1120    1   10    0    1
   3    0    8    1    0    0    0 1017    2    1
   0    0    5    0    0    0    9    2    2  992
   1    2    0  971    8    0    0    0    0    0
   2    3    1    1    2    0  879    0    1    3
   6  947    0    1    0    2    2    0    0    0
   0    0 1005    3    4    8    0    7    1    0
   4    0    2    0    3    0    4    1  958    2
   2    1    6   11  969    0    3    1   11    5

```

### Semi-supervised

```
python train.py -l 10
```