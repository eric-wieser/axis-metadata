import unittest
import numpy as np
from axis_metadata import ndarray as aarray
from axis_metadata import resolve_slice, broadcast_tuples, same_ignoring_nones

class TestEverything(unittest.TestCase):
	def test_resolve(self):
		self.assertSequenceEqual(
			list(resolve_slice(np.s_[0,...,3], 4)),
			list(zip(np.s_[0,:,:,3], np.arange(4)))
		)
		self.assertSequenceEqual(
			broadcast_tuples([(1,), (1, 2), (1, 2, 3)]), [(None, None, 1), (None, 1, 2), (1, 2, 3)]
		)

		self.assertEqual(same_ignoring_nones((1, None, 1, None)), 1)
		self.assertEqual(same_ignoring_nones((None, 1, None)), 1)
		with self.assertRaises(ValueError):
			same_ignoring_nones((None, 1, 2))

	def test_slicing(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])

		self.assertEqual(xa.axis_data, ('a', 'b', 'c'))
		self.assertEqual(xa[0].axis_data, ('b', 'c'))
		self.assertEqual(xa[:,0].axis_data, ('a', 'c'))
		self.assertEqual(xa[:,:,0].axis_data, ('a', 'b'))
		self.assertEqual(xa[...,0].axis_data, ('a', 'b'))
		self.assertEqual(xa[...,0,0,0].axis_data, ())

	def test_newaxis_slicing(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])

		self.assertEqual(xa[None].axis_data, (None, 'a', 'b', 'c'))
		self.assertEqual(xa[None,0].axis_data, (None,'b', 'c'))
		self.assertEqual(xa[:,None].axis_data, ('a', None, 'b', 'c'))
		self.assertEqual(xa[:,:,0, None].axis_data, ('a', 'b', None))
		self.assertEqual(xa[...,0].axis_data, ('a', 'b'))
		self.assertEqual(xa[None,...,0].axis_data, (None, 'a', 'b'))
		self.assertEqual(xa[...,None,0].axis_data, ('a', 'b', None))

	def test_transpose(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])
		self.assertEqual(xa.transpose().axis_data, ('c', 'b', 'a'))
		self.assertEqual(xa.T.axis_data, ('c', 'b', 'a'))

	def test_rollaxis(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])

		self.assertEqual(np.rollaxis(xa, -1).shape, (5, 2, 4))
		self.assertEqual(np.rollaxis(xa, -1).axis_data, ('c', 'a', 'b'))

	def test_ravel(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])

		self.assertEqual(xa.ravel().axis_data, (None,))

	def test_sum(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])
		self.assertEqual(xa.sum(axis=1).axis_data, ('a', 'c'))
		self.assertEqual(xa.sum(axis=(1,)).axis_data, ('a', 'c'))
		self.assertEqual(xa.sum(axis=(1,2)).axis_data, ('a',))
		self.assertEqual(xa.sum(axis=(1,), keepdims=True).axis_data, xa.axis_data)
		self.assertEqual(xa.sum(axis=(1,2), keepdims=True).axis_data, xa.axis_data)
		self.assertEqual(xa.sum(keepdims=True).axis_data, xa.axis_data)

	def test_add(self):
		x = np.empty((2, 4, 5))
		xa = aarray(x, ['a', 'b', 'c'])
		self.assertEqual((xa + xa).axis_data, xa.axis_data)

class TestSubclassing(unittest.TestCase):
	def test_it(self):
		class NamedAxisArray(aarray):
			def _resolve_axis(self, axis):
				if axis is not None and not isinstance(axis, int):
					axis = self.axis_data.index(axis)
				return axis

		x = np.empty((2, 4, 5))
		xa = NamedAxisArray(x, ['a', 'b', 'c'])
		self.assertEqual(xa.sum(axis='b').axis_data, ('a', 'c'))


unittest.main()
