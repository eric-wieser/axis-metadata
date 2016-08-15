import numpy as np

def resolve_slice(tup, ndim):
	# promote single indices to tuples
	if not isinstance(tup, tuple):
		tup = (tup,)

	# promote ellipsis to the right number of slices
	if Ellipsis in tup:
		if tup.count(Ellipsis) > 1:
			raise ValueError

		used_dim = sum(item not in (None, Ellipsis) for item in tup)
		ellipsis_width = ndim - used_dim
		if ellipsis_width < 0:
			raise ValueError

		at = tup.index(Ellipsis)
		tup = tup[:at] + (np.s_[:],) * ellipsis_width + tup[at+1:]

	# count non-inserted axes
	i = 0
	for key in tup:
		if key is np.newaxis:
			yield key, None
		else:
			yield key, i
			i = i + 1

	# add add trailing axes on the end
	while i < ndim:
		yield np.s_[:], i
		i = i + 1

def broadcast_tuples(ts):
	n = max(len(t) for t in ts)
	return [(None,) * (n - len(t)) + t for t in ts]

def same_ignoring_nones(ts):
	seen = None
	for t in ts:
		if t is not None:
			if seen is not None:
				if seen != t:
					raise ValueError
			else:
				seen = t

	return seen

class ndarray(np.ndarray):
	def __new__(cls, array, axis_data):
		assert len(axis_data) == array.ndim
		self = array.view(cls)
		self.axis_data = tuple(axis_data)
		return self

	def __getitem__(self, item):
		keep = tuple(
			self.axis_data[i]
			if i is not None else None
			for key, i in resolve_slice(item, self.ndim)
			if isinstance(key, slice) or i is None
		)
		res = super().__getitem__(item)
		if isinstance(res, ndarray):
			res.axis_data = keep
		return res

	def __array_finalize__(self, obj):
		if obj is None:
			return

		if isinstance(obj, ndarray):
			self.axis_data = obj.axis_data

		else:
			self.axis_data = (None,) * self.ndim

	def __array_prepare__(self, out_arr, context=None):
		out_arr = out_arr.view(type(self))

		if context is not None:
			func, args, domain = context
			data = [arg.axis_data for arg in args if isinstance(arg, ndarray)]
			out_data = list(out_arr.axis_data)
			for i, axis_items in enumerate(zip(*broadcast_tuples(data))):
				out_data[i] = same_ignoring_nones(axis_items)
			out_arr.axis_data = tuple(out_data)
		return out_arr

	def _resolve_axis(self, axis):
		return axis

	def sum(self, axis=None, dtype=None, out=None, keepdims=False):
		axis = self._resolve_axis(axis)
		res = super().sum(axis, dtype, out, keepdims)
		if out is None:
			res = res.view(ndarray)
		if isinstance(res, ndarray):
			if not keepdims:
				if isinstance(axis, int):
					axis = (axis,)
				if isinstance(axis, tuple):
					res.axis_data = tuple(self.axis_data[i] for i in range(self.ndim) if i not in axis)
				elif isinstance(axis, None):
					pass
				else:
					raise NotImplementedError
			else:
				res.axis_data = self.axis_data
		return res

	def transpose(self, *axes):
		if len(axes) == 0:
			axes = None
		elif len(axes) == 1 and not isinstance(axes[0], int):
			axes = axes[0]

		if axes is not None:
			axes = [self._resolve_axis(ax) for ax in axes]

		res = super().transpose(axes)
		if axes is None:
			res.axis_data = self.axis_data[::-1]
		else:
			res.axis_data = tuple(self.axis_data[i] for i in axes)
		return res

	@property
	def T(self):
	    return self.transpose() if self.ndim > 1 else self

	def reshape(self, *args, **kwargs):
		res = super().reshape(self, *args, **kwargs)
		res.axis_data = (None,) * res.ndim
		return res

	def ravel(self, *args, **kwargs):
		res = super().ravel(*args, **kwargs)
		res.axis_data = (None,)
		return res


