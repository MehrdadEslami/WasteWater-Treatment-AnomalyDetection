# class DataPreProcess:
#     def in
#
import shap
from keras.models import Sequential
# load your data here, e.g. X and y
# create and fit your model here

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
model = Sequential()
model.load('./model_conv1d.h5')


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

shap.summary_plot(shap_values, X, plot_type="bar")

# result_window_predict = {
#     "model": [],
#     "X_steps_window": [],
#     "y_predict": [],
#     'x_train.shape': [],
#     'x_test.shape': [],
#     'loss': [],
#     "train_acc": [],
#     "nit_acc": [],
#     "ph_acc": []
# }
# counter = 0
# for y_predict in range(2, 3):
#     for x_steps in range(1, 11):
#         print('The run =', counter)
#         counter = counter + 1;
#         result_window_predict['model'].append(counter)
#         result_window_predict["X_steps_window"].append(x_steps)
#         result_window_predict["y_predict"].append(y_predict)

# result_window_predict['x_train.shape'].append(x_train.shape)

# trainPredict = scaler.inverse_transform(trainPredict)
# testPredict = scaler.inverse_transform(testPredict)
#