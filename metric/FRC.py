import numpy as np
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd
import os
from tslearn.metrics import dtw
from functools import partial
import multiprocessing as mp


def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *,
             dtype=None):

    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=3)
    c = np.cov(x, y, rowvar, dtype=dtype)

    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]
    c = np.nan_to_num(c)

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)
    return c

def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> from sklearn.metrics import concordance_correlation_coefficient
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    if len(y_true.shape) >1:
        ccc_list = []
        for i in range(y_true.shape[1]):
            cor = corrcoef(y_true[:,i], y_pred[:, i])[0][1]
            mean_true = np.mean(y_true[:,i])

            mean_pred = np.mean(y_pred[:,i])


            var_true = np.var(y_true[:,i])
            var_pred = np.var(y_pred[:,i])

            sd_true = np.std(y_true[:,i])
            sd_pred = np.std(y_pred[:,i])

            numerator = 2 * cor * sd_true * sd_pred

            denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

            ccc = numerator / (denominator + 1e-8)

            ccc_list.append(ccc)
        ccc = np.mean(ccc_list)
    else:
        cor = np.corrcoef(y_true, y_pred)[0][1]
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)

        var_true = np.var(y_true)
        var_pred = np.var(y_pred)

        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)

        numerator = 2 * cor * sd_true * sd_pred

        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / (denominator + 1e-8)
    return ccc





def compute_FRC(args, pred, listener_em, val_test='val'):
    pred = pred
    listener_em = listener_em
    if val_test == 'val':
        speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    else:
        speaker_neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))

    all_FRC_list = []
    for i in range(pred.shape[1]):
        FRC_list = []
        for k in range(pred.shape[0]):
            speaker_neighbour_index = np.argwhere(speaker_neighbour_matrix[k] == 1).reshape(-1)
            speaker_neighbour_index_len = len(speaker_neighbour_index)
            ccc_list = []
            for n_index in range(speaker_neighbour_index_len):

                ''' 
                listener_em order :[listener1, listener2, listener3, ....., listener_n, speaker1, speaker2, speaker3, ....., speaker_n]
                listener1: [1, emotion_dim] 
                
                listener_em[speaker_neighbour_index[n_index]]: 
                1. speaker_neighbour_index[n_index]: speaker_j (with similar emotion as the speaker_k)
                2. listener_em[speaker_neighbour_index[n_index]]: emotion features of listener_j (speaker_j -> listener_j)
                So we can get an additional GT listener_j to listener_k (i.e., speaker_j -> listener_k)
                '''

                similar_listener_emotion = listener_em[speaker_neighbour_index[n_index]]
                ccc = concordance_correlation_coefficient(similar_listener_emotion.numpy(), pred[k,i].numpy())
                ccc_list.append(ccc)
            max_ccc = max(ccc_list)
            FRC_list.append(max_ccc)
        all_FRC_list.append(np.mean(FRC_list))
    return sum(all_FRC_list)



def _func(k_neighbour_matrix, k_pred, em=None):
    neighbour_index = np.argwhere(k_neighbour_matrix == 1).reshape(-1)
    neighbour_index_len = len(neighbour_index)
    max_ccc_sum = 0
    for i in range(k_pred.shape[0]):
        ccc_list = []
        for n_index in range(neighbour_index_len):
            emotion = em[neighbour_index[n_index]]
            ccc = concordance_correlation_coefficient(emotion, k_pred[i])
            ccc_list.append(ccc)
        max_ccc_sum += max(ccc_list)
    return max_ccc_sum



def compute_FRC_mp(args, pred, em, val_test='val', p=1):
    # pred: N 10 750 25
    # speaker: N 750 25
    if val_test == 'val':
        neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_val.npy'))
    else:
        neighbour_matrix = np.load(os.path.join(args.dataset_path, 'neighbour_emotion_test.npy'))

    FRC_list = []
    with mp.Pool(processes=p) as pool:
        # use map
        _func_partial = partial(_func, em=em.numpy())
        FRC_list += pool.starmap(_func_partial, zip(neighbour_matrix, pred.numpy()))
    return np.mean(FRC_list)
