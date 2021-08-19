import numpy as np
np.set_printoptions(threshold=np.inf)

Model_name = 'm_rnn'
saved_imputation = f'./result/{Model_name}_data.npy'
saved_label = f'./result/{Model_name}_label.npy'

imputated = np.load(saved_imputation)
label = np.load(saved_label)

print(imputated.shape)

imputated_reshape = np.array_split(imputated, imputated.shape[0]//6)

print(len(imputated_reshape))