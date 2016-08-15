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
		res.axis_data = keep
		return res

	def __array_finalize__(self, obj):
		if obj is None:
			return

		if isinstance(obj, ndarray):
			self.axis_data = obj.axis_data

		else:
			self.axis_data = (None,) * self.ndim

	def transpose(self, *axes):
		if len(axes) == 0:
			axes = None
		elif len(axes) == 1 and not isinstance(axes[0], int):
			axes = axes[0]

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


