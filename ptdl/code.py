# Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
# Đọc dữ liệu từ file CSV
df = pd.read_csv("data/student-por.csv")

#Tiền xử lý
#Lấy dl soos
df_numeric = df.select_dtypes(include=['number'])
#Xóa phần tử rỗng
df_clean = df_numeric.dropna()

#Đếm dl không bị khuyết
data_count = df_numeric.count()
print(data_count)
# Tính và in ra trung bình cộng theo hàng (axis=1)
row_means = df_numeric.mean(axis=1)
print("Trung bình cộng theo hàng:")
print(row_means)

# Tính và in ra trung bình cộng theo cột (axis=0)
column_means = df_numeric.mean(axis=0)
print("Trung bình cộng theo cột:")
print(column_means)

# Tính median của từng cột
column_medians = df_numeric.median()

print("Median của từng cột:")
print(column_medians)

# Tính mode của từng cột
column_modes = df_numeric.mode()

print("Mode của từng cột:")
print(column_modes)

# Tính giá trị max của từng cột
column_max = df_numeric.max()
print("Giá trị max của từng cột:")
print(column_max)

# Tính giá trị min của từng cột
column_min = df_numeric.min()
print("\nGiá trị min của từng cột:")
print(column_min)


# Tính Q1, Q2 , Q3 cho từng cột
column_q1 = df_numeric.quantile(0.25)
column_q2 = df_numeric.median()
column_q3 = df_numeric.quantile(0.75)
column_IQR = column_q3 - column_q1
print("Q1 của từng cột:")
print(column_q1)

print("\nMedian của từng cột:")
print(column_q2)

print("\nQ3 của từng cột:")
print(column_q3)

print("\nIQR của từng cột:")
print(column_IQR)

# Tính phương sai của từng cột
column_variances = df_numeric.var()
print("Phương sai của từng cột:")
print(column_variances)

# Tính độ lệch chuẩn của từng cột
column_std_devs = df_numeric.std()
print("\nĐộ lệch chuẩn của từng cột:")
print(column_std_devs)


def descriptive(data_count, column_min, column_max, column_medians, column_modes, column_q1, column_q2, column_q3,
                column_IQR, column_variances, column_std_devs):
    data = {'Count': [i for i in data_count],
            'min': [i for i in column_min],
            'max': [i for i in column_max],
            'median': [i for i in column_medians],
            'mode': [i for i in column_modes.values[0]],
            'Q1': [i for i in column_q1],
            'Q2': [i for i in column_q2],
            'Q3': [i for i in column_q3],
            'IQR': [i for i in column_IQR],
            'Variance': [i for i in column_variances],
            'stdev': [i for i in column_std_devs],
            }  # dữ liệu đang ở dạng dic
    df1 = pd.DataFrame(data)  # convert về dạng pandas
    df1.index = df_numeric.keys()  # keys sẽ trả về tên của các cột( features)
    data_complete = df1.transpose()  # transpose để chuyển hàng về cột, cột về hàng

    # Thêm một cột mới vào đầu DataFrame
    new_column_data = ['count', 'min', 'max', 'median', 'mode', 'Q1', 'Q2', 'Q3', 'IQR', 'Variance', 'stdev']
    column_name = ' '
    data_complete.insert(loc=0, column=column_name, value=new_column_data)
    print(data_complete.to_string())
    data_complete.to_csv('Data/Thong_ke_1.txt', sep='\t', index=False)


descriptive(data_count, column_min, column_max, column_medians, column_modes, column_q1, column_q2, column_q3,
            column_IQR, column_variances, column_std_devs)
print(
    '---------------------------------------------------------------------------------------------------------------------------------------------')

correlation_matrix = df_numeric[['G1', 'G2', 'G3', 'goout', 'Dalc', 'freetime', 'failures', 'studytime', 'traveltime']]

print(correlation_matrix.corr().to_string())


predict = 'G3'
x = np.array(correlation_matrix.drop([predict],axis=1))
y = np.array(correlation_matrix[predict])

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size = 0.2,random_state = 20)
regressor = LinearRegression()
regressor.fit(x_train,y_train)

predictions = regressor.predict(x_test)

# Tính toán RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Tính toán MAE
mae = mean_absolute_error(y_test, predictions)

# Tính toán R-squared
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)

print("G3_true\tG3_pred")
for x in range(len(predictions)):
    print(round(predictions[x]), "\t", y_test[x])


errors = y_test - predictions

# Vẽ đồ thị boxplot
plt.boxplot(errors)

plt.xlabel('Dự đoán')
plt.ylabel('Sai số')
plt.title('Boxplot - Sai số dự đoán')
plt.show()

# Vẽ đồ thị histogram
plt.hist(errors, bins=20)

plt.xlabel('Sai số')
plt.ylabel('Số lượng')
plt.title('Phân bố sai số dự đoán')
plt.show()

# Vẽ đồ thị scatter plot
plt.scatter(y_test, predictions, color='blue', label='Thực tế vs. Dự đoán')

plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Biểu đồ scatter - Thực tế vs. Dự đoán')
plt.legend()
plt.show()

correlation_matrix.boxplot(column=['G1', 'G2', 'goout', 'Dalc', 'freetime', 'failures', 'studytime', 'traveltime', 'G3'])

plt.xlabel('Biến')
plt.ylabel('Giá trị')
plt.title('Biểu đồ boxplot - Phụ thuộc G3 vào các biến')
plt.show()

# Vẽ đồ thị histogram
correlation_matrix.hist(column=['G1', 'G2', 'goout', 'Dalc', 'freetime', 'failures', 'studytime', 'traveltime', 'G3'], bins=20)

plt.xlabel('Giá trị')
plt.ylabel('Số lượng')
plt.title('Biểu đồ histogram - Phụ thuộc G3 vào các biến')
plt.show()

# Vẽ đồ thị scatter plot
plt.scatter(correlation_matrix['G1'], correlation_matrix['G3'], color='blue', label='G1')
plt.scatter(correlation_matrix['G2'], correlation_matrix['G3'], color='red', label='G2')

plt.xlabel('G1 và G2')
plt.ylabel('G3')
plt.title('Biểu đồ scatter - G3 phụ thuộc vào G1 và G2')
plt.legend()
plt.show()
