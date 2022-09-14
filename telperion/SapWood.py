class SapWood:
	def __init__(self, zero=0, one=1):
		self.zero = zero
		self.one  = one
		
	def fit(self, x, y):
		device = "cuda" if torch.cuda.is_available() else "cpu"
		print(f"Using {device} device")
		
	def predict(self, x):
		pass
		# y_hat = self.__heartwood(x)
		
		# if y_hat == 0:
		# 	return self.zero
		# else:
		# 	return self.one