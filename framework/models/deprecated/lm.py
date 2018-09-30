#Implementation of simple linear regression 

class LM():
	def __init__(self, x, y):
		self._x = x
		self._y = y
		self._x_mean = sum(self._x)/len(self._x)
		self._y_mean = sum(self._y)/len(self._y)

		self._beta = self.calc_beta()
		if self._beta:
			self._alpha = self._y_mean - (self._beta * self._x_mean)
		else:
			self._alpha = self._y_mean

	def calc_beta(self):
		s_xx = sum(map(lambda a: (a - self._x_mean)**2, self._x))

		s_xy = sum([(self._x[i] - self._x_mean) * (self._y[i] - self._y_mean) for i in range(len(self._x))])

		if (s_xx):
			return (s_xy/s_xx)
		else:
			return None
