import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
"""

result = {
    "model": [0, 1, 2, 3, 4, 5],
    "X_steps": [7, 7, 7, 7, 7, 7],
    "y_steps": [1, 1, 1, 1, 1, 1],
    'x_train.shape': [(820, 7, 7), (820, 7, 7), (820, 7, 7), (820, 7, 7), (820, 7, 7), (820, 7, 7)],
    'x_train.shape': [(201, 7, 7), (201, 7, 7), (201, 7, 7), (201, 7, 7), (201, 7, 7), (201, 7, 7)],
    "normal": ['MinMax', 'MinMax', 'Standard', 'Standard', 'Standard', 'Standard'],
    "model": ['l50_D2', 'l50_D2', 'l50_D2', 'l50_l100_D2', 'l50_l100_D2', 'l50_l100_l50_D2'],
    "optimizer": ['adam', 'RMSprop', 'RMSprop', 'RMSprop', 'adam', 'RMSprop'],
    "epoch": [100, 100, 100, 100, 100, 100],
    "batch_size": [1, 1, 1, 1, 1, 1],
    'loss': [None, None, 0.0174, 0.0152, 0.0136, 0.013],
    "train_acc": [0.7817, 0.7866, 0.95, 0.9623, 0.958, 0.97],
    "nit_acc": [24.05, 23.58, 1.51, 1.1, 1.09, 0.98],
    "ph_acc": [0.91, 1.64, 1.11, 0.1, 0.83, 0.89]
}

result_window_predict0 = {
    "model": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "X_steps_window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3],
    "y_predict": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    'x_train.shape': [(826, 1, 7), (825, 2, 7), (824, 3, 7), (823, 4, 7), (822, 5, 7), (821, 6, 7), (820, 7, 7), (819, 8, 7), (818, 9, 7), (817, 10, 7), (826, 1, 7), (825, 2, 7), (824, 3, 7)],
    'x_test.shape': [(207, 1, 7), (206, 2, 7), (205, 3, 7), (204, 4, 7), (203, 5, 7), (202, 6, 7), (201, 7, 7), (200, 8, 7), (199, 9, 7), (198, 10, 7), (207, 1, 7), (206, 2, 7), (205, 3, 7)],
    'loss': [0.7528, 0.0340, 0.0153, 0.0131, 0.0136, 0.0125, 0.0127, 0.0138, 0.0157, 0.0137, 0.7580, 0.0383, 0.0151],
    "train_acc": [0.6755, 0.9539, 0.9684, 0.9647, 1.0841, 0.9562, 0.9585, 0.9548, 0.9560, 0.9608, 0.6441, 0.9358, 0.9624],
    "nit_acc": [0.9639, 1.3530, 1.1888, 1.1310, 0.9260, 1.0742, 1.0764, 1.1883, 1.2970, 1.3073, 1.0701, 1.4448, 1.3504],
    "ph_acc": [0.7818, 1.0556, 1.0387, 0.9747, 0.1186, 0.8620, 0.9105, 0.9671, 1.0970, 0.9810, 0.9288, 1.1919, 1.1020]
}

result_window_predict3 = {
    "model": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    "X_steps_window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y_predict": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'x_train.shape': [(826, 1, 7), (825, 2, 7), (824, 3, 7), (823, 4, 7), (822, 5, 7), (821, 6, 7), (820, 7, 7), (819, 8, 7), (818, 9, 7), (817, 10, 7)],
    'x_test.shape': [(207, 1, 7), (206, 2, 7), (205, 3, 7), (204, 4, 7), (203, 5, 7), (202, 6, 7), (201, 7, 7), (200, 8, 7), (199, 9, 7), (198, 10, 7)],
    'loss': [0.8707, 0.0664, 0.0210, 0.0192, 0.0183, 0.0196, 0.0175, 0.0193, 0.0213, 0.0208],
    "train_acc": [0.6356, 0.9188, 0.9502, 0.9550, 0.9574, 0.9562, 0.9488, 0.9512, 0.9560, 0.9437],
    "nit_acc": [1.0820, 1.5159, 1.2896, 1.1908, 1.1839, 1.3690, 1.2147,  1.4110, 1.2206, 1.0574],
    "ph_acc": [0.8432, 1.2869, 1.0101, 1.0934, 0.9057, 0.9415, 1.0646, 1.1178, 1.1356, 1.0795]
}

result_window_predict03 = {
    "model": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    "X_steps_window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y_predict": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'x_train.shape': [(826, 1, 7), (825, 2, 7), (824, 3, 7), (823, 4, 7), (822, 5, 7), (821, 6, 7), (820, 7, 7), (819, 8, 7), (818, 9, 7), (817, 10, 7), (826, 1, 7), (825, 2, 7), (824, 3, 7), (823, 4, 7), (822, 5, 7), (821, 6, 7), (820, 7, 7), (819, 8, 7), (818, 9, 7), (817, 10, 7)],
    'x_test.shape': [(207, 1, 7), (206, 2, 7), (205, 3, 7), (204, 4, 7), (203, 5, 7), (202, 6, 7), (201, 7, 7), (200, 8, 7), (199, 9, 7), (198, 10, 7),  (207, 1, 7), (206, 2, 7), (205, 3, 7), (204, 4, 7), (203, 5, 7), (202, 6, 7), (201, 7, 7), (200, 8, 7), (199, 9, 7), (198, 10, 7)],
    'loss': [0.7528, 0.0340, 0.0153, 0.0131, 0.0136, 0.0125, 0.0127, 0.0138, 0.0157, 0.0137,  0.8707, 0.0664, 0.0210, 0.0192, 0.0183, 0.0196, 0.0175, 0.0193, 0.0213, 0.0208],
    "accuracy": [0.6755, 0.9539, 0.9684, 0.9647, 1.0841, 0.9562, 0.9585, 0.9548, 0.9560, 0.9608, 0.6356, 0.9188, 0.9502, 0.9550, 0.9574, 0.9562, 0.9488, 0.9512, 0.9560, 0.9437],
    "nit_acc": [0.9639, 1.3530, 1.1888, 1.1310, 0.9260, 1.0742, 1.0764, 1.1883, 1.2970, 1.3073, 1.0820, 1.5159, 1.2896, 1.1908, 1.1839, 1.3690, 1.2147,  1.4110, 1.2206, 1.0574],
    "ph_acc": [0.7818, 1.0556, 1.0387, 0.9747, 0.1186, 0.8620, 0.9105, 0.9671, 1.0970, 0.9810, 0.8432, 1.2869, 1.0101, 1.0934, 0.9057, 0.9415, 1.0646, 1.1178, 1.1356, 1.0795]
}
df0 = pd.DataFrame(result_window_predict0)
df3 = pd.DataFrame(result_window_predict3)
df1 = pd.read_csv('1_10_edit.csv')
df2 = pd.read_csv('2_10_edit.csv')
df7 = pd.read_csv('7_10_edit.csv')
df_merge = pd.concat([df0,df1,df2, df3, df7])
df_merge.to_csv('merge_final.csv')
"""


"""Convert """
# for i in range(0, df.shape[0]):
#     df['loss'][i] = round(df['loss'][i],4)
#     df['train_acc'][i] = round(df['train_acc'][i], 4)
#     df['nit_acc'][i] = round(df['nit_acc'][i], 4)
#     df['ph_acc'][i] = round(df['ph_acc'][i], 4)
#
# df.to_csv('merge_final_round.csv')
# for i in range(0, 10):
#     t1 = list(df['loss'][i][1:-1].split(","))
#     t2 = list(df['train_acc'][i][1:-1].split(","))
#     t1 = [float(x) for x in t1]
#     t2 = [float(x) for x in t2]
#     df['loss'][i] = min(t1)
#     df['train_acc'][i] = max(t2)
#
# df.to_csv('2_10_edit.csv')
# result_df = pd.DataFrame(result_window_predict)
# result_df3 = pd.DataFrame(result_window_predict3)
#
# # print(result_df)
# # print(result_df2)
#

"""HEATMAP NITrate loss"""
# df = pd.read_csv('merge_final_round.csv')
# print(df.shape)
# headmap_dict = {
#     'windows_Size': df['X_steps_window'],
#     'prediction_Horizon': df['y_predict'],
#     'nitrate_Accuracy': df['nit_acc']
# }
# headmap_df = pd.DataFrame(headmap_dict)
# v_min = min(df['nit_acc'])
# v_max = max(df['nit_acc'])
# np_nit = np.array(df["nit_acc"])
# # print(np_nit.shape)
# # np_nit.reshape(8, 10)
# summary = pd.pivot_table(data=headmap_df, index='prediction_Horizon', columns='windows_Size' , values='nitrate_Accuracy')
# # summary = pd.pivot_table(data=summary, index='y_steps', columns='nit_acc', values='nit_acc')
# sns.set()
# ax = sns.heatmap(summary, vmin=0.8, vmax=1.6)
# plt.title('Nitrate Loss')
# plt.show()

"""HEATMAP NITrate loss"""

# df = pd.read_csv('merge_final_round.csv')
# print(df.shape)
# headmap_dict = {
#     'windows_Size': df['X_steps_window'],
#     'prediction_Horizon': df['y_predict'],
#     'Phosphate_Accuracy': df['ph_acc']
# }
# headmap_df = pd.DataFrame(headmap_dict)
# v_min = min(df['ph_acc'])
# v_max = max(df['ph_acc'])
# np_nit = np.array(df["nit_acc"])
# # print(np_nit.shape)
# # np_nit.reshape(8, 10)
# summary = pd.pivot_table(data=headmap_df, index='prediction_Horizon', columns='windows_Size' , values='Phosphate_Accuracy')
# # summary = pd.pivot_table(data=summary, index='y_steps', columns='nit_acc', values='nit_acc')
# sns.set()
# ax = sns.heatmap(summary, vmin=v_min-0.05, vmax=v_max+0.05, cmap='coolwarm')
# plt.title('Phosphate Loss')
# plt.show()


# plt.plot()
# x = np.array(["Phosphat without", "Phosphat with", "Nitrate without", "Nitrate with"])
# y = np.array([0.1124, 0.0945, 0.1651, 0.1213])
# plt.title('Prediction Loss With/Without Climate Data')
# plt.bar(x,y, color=['yellow','green', 'yellow','green'] , width=0.1)
# plt.ylabel('loss')
# plt.show()

x = np.arange(-10,10, 0.01)
# y = np.random.rand(10)
y = 1/(1+np.exp(-1*x))
y2 = []
for i in range(0,len(y2)):
    if y[i] > 0.5:
        y2.append(1)
    else:
        y2.append(0)
plt.plot(x,np.log(y))
plt.show()