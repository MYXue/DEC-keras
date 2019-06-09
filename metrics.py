import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score #互信息
ari = adjusted_rand_score #标准兰德系数


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    # 线性分配问题（linear assignment problem）
    # N个人分配N项任务，一个人只能分配一项任务，一项任务只能分配给一个人，
    # 将一项任务分配给一个人是需要支付报酬，如何分配任务，保证支付的报酬总数最小。
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w) # 因为原函数的目标是为了使报酬最小，所以此处做了一个转化w.max() - w
    # linear_assignment的返回值是array,The pairs of (row, col) indices in the original array giving the original ordering.
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size