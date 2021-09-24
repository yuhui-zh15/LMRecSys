from collections import Counter
import numpy as np
import json
import random


def compute_metrics(top_preds, labels, prefix):
    # top_preds: (B x N), labels: (B)
    r_20 = compute_recall_at_k(top_preds, labels, k=20)
    mrr_20 = compute_mrr_at_k(top_preds, labels, k=20)
    ndcg_20 = compute_ndcg_at_k(top_preds, labels, k=20)
    return {f'{prefix}/r@20': r_20, f'{prefix}/mrr@20': mrr_20, f'{prefix}/ndcg@20': ndcg_20}


def compute_precision_at_k(top_preds, labels, k=20):
    assert(len(top_preds) == len(labels))
    r_k = (top_preds[:, :k] == labels.reshape(-1, 1)).mean(-1).mean(-1)
    return float(r_k)

def compute_recall_at_k(top_preds, labels, k=20):
    assert(len(top_preds) == len(labels))
    r_k = (top_preds[:, :k] == labels.reshape(-1, 1)).sum(-1).mean(-1)
    return float(r_k)

def compute_mrr_at_k(top_preds, labels, k=20):
    assert(len(top_preds) == len(labels))
    top_preds, labels = top_preds[:, :k].tolist(), labels.tolist()
    mrr_k = np.mean([1 / (top_pred.index(label) + 1) if label in top_pred else 0. for top_pred, label in zip(top_preds, labels)])
    return float(mrr_k)

def compute_ndcg_at_k(top_preds, labels, k=20):
    assert(len(top_preds) == len(labels))
    rs = (top_preds == labels.reshape(-1, 1)).astype(int).tolist()
    ndcg_k = np.mean([ndcg_at_k(r, k) for r in rs])
    return float(ndcg_k)


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if np.size(r):
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, np.size(r) + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, np.size(r) + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def assert_equal(x, y, eps=1e-6):
    return -eps < x - y < eps 


def metric_unit_tests():
    top_preds = np.array([[1,2,3], [4,5,6], [7,8,9]])
    labels = np.array([3,5,7])

    mrr_at_1 = compute_mrr_at_k(top_preds, labels, k=1)
    assert_equal(mrr_at_1, (0+0+1/1)/3)
    mrr_at_2 = compute_mrr_at_k(top_preds, labels, k=2)
    assert_equal(mrr_at_1, (0+1/2+1/1)/3)
    mrr_at_3 = compute_mrr_at_k(top_preds, labels, k=3)
    assert_equal(mrr_at_1, (1/3+1/2+1/1)/3)
    
    recall_at_1 = compute_recall_at_k(top_preds, labels, k=1)
    assert_equal(recall_at_1, 1/3)
    recall_at_2 = compute_recall_at_k(top_preds, labels, k=2)
    assert_equal(recall_at_2, 2/3)
    recall_at_3 = compute_recall_at_k(top_preds, labels, k=3)
    assert_equal(recall_at_3, 3/3)
    
    precision_at_1 = compute_precision_at_k(top_preds, labels, k=1)
    assert_equal(precision_at_1, (0/1+0/1+1/1)/3)
    precision_at_2 = compute_precision_at_k(top_preds, labels, k=2)
    assert_equal(precision_at_2, (0/2+1/2+1/2)/3)
    precision_at_3 = compute_precision_at_k(top_preds, labels, k=3)
    assert_equal(precision_at_3, (1/3+1/3+1/3)/3)
    
    # TODO: not sure if the assertion is correct...
    ndcg_at_1 = compute_ndcg_at_k(top_preds, labels, k=1)
    assert_equal(ndcg_at_1, 0.3333333333333333)
    ndcg_at_2 = compute_ndcg_at_k(top_preds, labels, k=2)
    assert_equal(ndcg_at_2, 0.6666666666666666)
    ndcg_at_3 = compute_ndcg_at_k(top_preds, labels, k=3)
    assert_equal(ndcg_at_3, 0.8769765845238192)


def random_baseline(data, n_item, k=20):
    top_preds = np.array([random.sample(list(range(n_item)), k) for i in range(len(data))])
    val_labels = np.array([item['ids'][-2] for item in data])
    test_labels = np.array([item['ids'][-1] for item in data])
    print(compute_metrics(top_preds, val_labels, 'val'))
    print(compute_metrics(top_preds, test_labels, 'test'))


def maxfreq_baseline(data):
    train_ids = [id for item in data for id in item['ids'][:-2]]
    most_freq_ids, _ = zip(*Counter(train_ids).most_common())
    top_preds = np.array([[id for id in most_freq_ids] for i in range(len(data))])
    val_labels = np.array([item['ids'][-2] for item in data])
    test_labels = np.array([item['ids'][-1] for item in data])
    print(compute_metrics(top_preds, val_labels, 'val'))
    print(compute_metrics(top_preds, test_labels, 'test'))
    

if __name__ == '__main__':
    metric_unit_tests()
    data = [json.loads(line) for line in open('datasets/MovieLens-1M-5Star/data.jsonl')]
    for item in data: item['ids'] = item['ids'][-4:]
    # random_baseline(data, 3883, 20)
    maxfreq_baseline(data)