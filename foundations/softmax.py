import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxNum = np.max(z)
        eArray = np.exp(z-maxNum)
        eSum = np.sum(eArray)

        return np.round(eArray/eSum,4)