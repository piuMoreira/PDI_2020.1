import numpy as np
import util as imgutil


def templatematching(template, image):
    template -= min(np.min(template), 0)
    image -= min(np.min(image), 0)
    extimage = imgutil.extend_with_zeros_mask(image, np.rot90(template).shape)

    cross_corr = imgutil.apply_mask(extimage, np.flip(np.rot90(template, 2)))
    m, n = template.shape
    mn = m * n

    def ls(A, m, n):
        B = np.pad(A, ((m, m), (n, n)))
        s = np.cumsum(B, axis=0)
        c = s[m:-1, :] - s[:-m - 1, :]
        s = np.cumsum(c, axis=1)
        return s[:, n:-1] - s[:, :-n - 1]

    lsA2 = ls(image ** 2, m, n)
    lsA = ls(image, m, n)

    diff_local_sums = (lsA2 - (lsA ** 2) / mn)
    denom_A = np.sqrt(np.maximum(diff_local_sums, 0))

    denom_T = np.sqrt(mn - 1) * np.std(template, ddof=1)
    denom = denom_T * denom_A
    numerator = cross_corr - lsA * template.sum() / mn

    out = np.zeros(numerator.shape)
    tol = np.sqrt(np.spacing(np.max(np.abs(denom))))
    i_nonzero = np.where(denom > tol)
    out[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]

    out[np.where((np.abs(out) - 1) > np.sqrt(np.spacing(1)))] = 0
    return out
