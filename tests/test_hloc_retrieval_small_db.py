import unittest

import numpy as np
import torch

from hloc.pairs_from_retrieval import pairs_from_score_matrix


class RetrievalSmallDatabaseTest(unittest.TestCase):
    def test_request_larger_than_database_keeps_only_valid_pairs(self):
        scores = torch.tensor([[.1, .9, .4], [.8, .2, .7]])
        invalid = np.array([[False, True, False], [False, False, True]])
        pairs = pairs_from_score_matrix(scores.clone(), invalid.copy(), 20, min_score=.15)
        self.assertEqual(pairs, [(0, 2), (1, 0), (1, 1)])

    def test_numpy_scores_empty_database_and_zero_selection(self):
        scores = np.array([[4., 2.]])
        self.assertEqual(pairs_from_score_matrix(scores.copy(), np.zeros((1, 2), bool), 0), [])
        empty = torch.empty((2, 0))
        self.assertEqual(pairs_from_score_matrix(empty, np.empty((2, 0), bool), 5), [])

    def test_small_k_matches_independent_sorting(self):
        scores = np.array([[.2, .8, .4, .1], [.5, .3, .9, .7]], dtype=np.float32)
        invalid = np.array([[False, False, True, False], [True, False, False, False]])
        for k in [1, 2, 4, 10]:
            expected = []
            for i, row in enumerate(scores):
                eligible = [j for j in range(len(row)) if not invalid[i, j] and row[j] >= .25]
                eligible.sort(key=lambda j: row[j], reverse=True)
                expected.extend((i, j) for j in eligible[:k])
            self.assertEqual(pairs_from_score_matrix(scores.copy(), invalid.copy(), k, .25), expected)


if __name__ == "__main__":
    unittest.main()
