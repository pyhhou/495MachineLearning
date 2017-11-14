from helper import *

X, y = read_data('wavy_data.csv')
num_fold, num_degree = 3, 8
folds = random_split(P=y.size, K=num_fold)

X_train, y_train, X_val, y_val = train_val_split(X, y, folds, fold_id=0)

MSE_train, MSE_val = [], []
D = np.arange(1, num_degree+1)
for d in D:
    F_train = fourier_basis(X_train, D=d)
    F_val = fourier_basis(X_val, D=d)
    w = least_square_sol(F_train, y_train)
    MSE_train.append(mean_square_error(w, F_train, y_train))
    MSE_val.append(mean_square_error(w, F_val, y_val))

print 'The best degree of Fourier basis, in terms of validation error, is %d' % (MSE_val.index(min(MSE_val))+1)
make_plot(D, MSE_train, MSE_val)