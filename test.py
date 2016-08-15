import unittest
import numpy as np
from axis_metadata import ndarray as aarray
from axis_metadata import resolve_slice

class TestEverything(unittest.TestCase):
	def test_resolve(self):

		self.assertSequenceEqual(
			list(resolve_slice(np.s_[0,...,3], 4)),
			list(zip(np.s_[0,:,:,3], np.arange(4)))
		)

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

unittest.main()
