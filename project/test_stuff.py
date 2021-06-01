# generate dataset
X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# define feature selection
fs = SelectKBest(score_func=f_regression, k=1000)
# apply feature selection
X_selected = fs.fit_transform(X, y)
print(X_selected.shape)

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))


# data = np.random.standard_normal(size = (4,3)) # Use np.random.standard_normal instead of np.random.randn
# display(data)
# df = pd.DataFrame(data)
# display(df)

# simple linear regression - data definition
# x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# y = np.array([5, 20, 14, 32, 22, 38])

# multiple linear regression - data definition
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

# Training the model
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_) # w0 or b0
print('slope:', model.coef_) # w1, w2...

y_pred = model.predict(x) # test it against the input (this is will tell us how well it works on the data set it trained on - not a good metric)
print('predicted response:', y_pred, sep='\n')

# polynomial regression - data definition

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
model = LinearRegression().fit(x_, y)

r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

y_pred = model.predict(x_)
print('predicted response:', y_pred, sep='\n')