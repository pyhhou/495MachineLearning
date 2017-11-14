from helper import *


X, y = read_data('galileo_ramp_data.csv')
num_fold, num_degree = 6, 6
folds = random_split(P=y.size, K=num_fold)

MSE_train, MSE_val = [0]*num_degree, [0]*num_degree
D = np.arange(1, num_degree+1)
for f in xrange(num_fold):
	X_train, y_train, X_val, y_val = train_val_split(X, y, folds, fold_id=f)
	for i, d in enumerate(D):
		F_train = poly_basis(X_train, D=d)
		F_val = poly_basis(X_val, D=d)
		w = least_square_sol(F_train, y_train)
		MSE_train[i] += mean_square_error(w, F_train, y_train)/num_fold
		MSE_val[i] += mean_square_error(w, F_val, y_val)/num_fold

print 'The best degree of polynomial basis, in terms of validation error, is %d' % (MSE_val.index(min(MSE_val))+1)

make_plot(D, MSE_train, MSE_val)