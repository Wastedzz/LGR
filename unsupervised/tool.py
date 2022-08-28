import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# def pd_toExcel(data, fileName):  # pandas库储存数据到excel
#
#     EPOCH = []
#     df = pd.DataFrame(data)  # 创建DataFrame
#     df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）
#
#
# def smooth(data, sm=1):
#     smooth_data = []
#     if sm > 0.5:
#         for d in data:
#             z = np.ones(len(d))
#             y = np.ones(sm) * 1.0
#             d = np.convolve(y, d, "same") / np.convolve(y, z, "same")
#             smooth_data.append(d)
#     return smooth_data


def excel_read_plt(filename):
    plt.figure()
    for file in filename:
        pd_data = pd.read_excel(file)
        data = pd_data.values
        plt.plot(data[:, 0], data[:, 1])
    # pd_data2 = pd.read_excel('info_nce_10.xlsx')
    # data2 = pd_data2.values
    # sm_data2 = smooth(data2)
    # sm_data2 = np.array(sm_data2)
    # plt.plot(data1[:, 0], data1[:, 1])
    # plt.plot(data2[:, 0], sm_data2[:, 1])
    # plt.ylim((4, 9))
    plt.legend(labels=['InfoGraph', 'Ours'], prop={'family': 'Times New Roman', 'size': 14})
    plt.xlabel('Epoch', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('NCE', fontdict={'family': 'Times New Roman', 'size': 16})

    plt.savefig('infograph1.pdf')
    plt.savefig(r'C:\Users\ztrh\Desktop\Projects\Graph文章\m1.png')
    plt.show()
    return data


filenames = ['info_nce.xlsx', 'info_nce_16.xlsx']
file_name = excel_read_plt(filenames)
