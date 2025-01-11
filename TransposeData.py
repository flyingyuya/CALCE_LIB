import numpy as np
import pandas as pd
import glob

def drop_outlier(array,count,bins):
    index = []
    range_ = np.arange(1,count,bins)
    for i in range_[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)

# 加载数据
def load_data(Battary_list, dir_path):
    Battery = {}
    # 遍历每个电池数据
    for name in Battary_list:
        print('Load Dataset ' + name + ' ...')
        # 查找指定路径下的所有 excel 文件
        path = glob.glob(dir_path + name + '/*.xlsx')
        dates = []
        # 遍历每个文件
        for p in path:
            # 读取文件
            print('Load ' + str(p) + ' ...')
            df = pd.read_excel(p, sheet_name=1, engine='openpyxl')
            
            # 保存日期列中的第一个日期
            dates.append(df['Date_Time'][0])
        # 对每个文件按第一个日期进行排序，返回排序后的索引
        idx = np.argsort(dates)
        # 使用排序后的索引得到排序后的文件路径的数组
        path_sorted = np.array(path)[idx]

        count = 0
        discharge_capacities = []
        health_indicator = []
        internal_resistance = []
        CCCT = []
        CVCT = []
        Dis_Current = []
        Dis_Voltage = []
        Dis_Time = []
        # 遍历排序后的每个文件
        for p in path_sorted:
            df = pd.read_excel(p, sheet_name=1, engine='openpyxl')
            print('Load ' + str(p) + ' ...')
            # 对 Cycle_Index 列去重后转为列表
            cycles = list(set(df['Cycle_Index']))
            # 遍历每个周期
            for c in cycles:
                # 得到一个周期内的所有数据
                df_lim = df[df['Cycle_Index'] == c]
                # Charging data
                df_c = df_lim[(df_lim['Step_Index'] == 2)|(df_lim['Step_Index'] == 4)]
                c_v = df_c['Voltage(V)']
                c_c = df_c['Current(A)']
                c_t = df_c['Test_Time(s)']
                # 获取恒流充电(CC)和恒压充电(CV)阶段的数据
                df_cc = df_lim[df_lim['Step_Index'] == 2]  # 恒流充电阶段
                df_cv = df_lim[df_lim['Step_Index'] == 4]  # 恒压充电阶段
                # 计算恒流充电时间
                CCCT.append(np.max(df_cc['Test_Time(s)'])-np.min(df_cc['Test_Time(s)']))
                # 计算恒压充电时间
                CVCT.append(np.max(df_cv['Test_Time(s)'])-np.min(df_cv['Test_Time(s)']))

                # Discharging data
                df_d = df_lim[df_lim['Step_Index'] == 7]
                d_v = df_d['Voltage(V)']
                d_c = df_d['Current(A)']
                d_t = df_d['Test_Time(s)']
                d_im = df_d['Internal_Resistance(Ohm)']

                if(len(list(d_c)) != 0):
                    # 计算相邻时间点的差值
                    time_diff = np.diff(list(d_t))
                    # 取电流数据(去掉第一个点)
                    d_c = np.array(list(d_c))[1:]
                    # 计算放电容量 Q = I*t (Ah)
                    discharge_capacity = time_diff*d_c/3600 
                    # 计算累积放电容量
                    discharge_capacity = [np.sum(discharge_capacity[:n]) 
                                          for n in range(discharge_capacity.shape[0])]
                    # 保存最终放电容量(取负值)
                    discharge_capacities.append(-1*discharge_capacity[-1])

                    # 计算电压在3.8V和3.4V时对应的容量差值作为健康指标
                    dec = np.abs(np.array(d_v) - 3.8)[1:] # 找到最接近3.8V的点
                    start = np.array(discharge_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(d_v) - 3.4)[1:] # 找到最接近3.4V的点
                    end = np.array(discharge_capacity)[np.argmin(dec)]
                    health_indicator.append(-1 * (end - start))

                    # 计算平均内阻
                    internal_resistance.append(np.mean(np.array(d_im)))
                    # 保存充放电数据
                    Dis_Current.append(d_c)
                    Dis_Voltage.append(np.array(list(d_v[1:])))
                    Dis_Time.append(np.array(list(d_t[1:])))
                    count += 1

        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)

        idx = drop_outlier(discharge_capacities, count, 40)
        df_result = pd.DataFrame({
            'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
            'capacity': discharge_capacities[idx],
            'SoH': health_indicator[idx],
            'resistance': internal_resistance[idx],
            'Dis_Current': [Dis_Current[i] for i in idx],
            'Dis_Voltage': [Dis_Voltage[i] for i in idx],
            'Dis_Time': [Dis_Time[i] for i in idx],
            'CCCT': CCCT[idx],
            'CVCT': CVCT[idx]
        })
        Battery[name] = df_result
    return Battery

Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
dir_path = r'BatteryDataset/'

# 加载并处理数据
Battery = load_data(Battery_list, dir_path)

# 将数据保存为npy格式
np.save(dir_path + 'CALCE.npy', Battery)
print('数据已成功保存到 CALCE.npy')

# 读取保存好的数据
Battery = np.load(dir_path + 'CALCE.npy', allow_pickle=True)
Battery = Battery.item()

# 创建一个空的DataFrame来存储所有电池的数据
all_battery_data = pd.DataFrame()

# 首先获取最大循环次数,用于对齐数据
max_cycles = 0
for battery_name in Battery_list:
    cycles = len(Battery[battery_name]['cycle'])
    if cycles > max_cycles:
        max_cycles = cycles

# 创建统一的cycle列
all_battery_data['cycle'] = range(1, max_cycles + 1)

# 为每个电池添加容量数据列
for battery_name in Battery_list:
    battery_cycles = Battery[battery_name]['cycle']
    battery_capacity = Battery[battery_name]['capacity']
    
    # 直接使用原始列表数据
    battery_Dis_Current = Battery[battery_name]['Dis_Current']
    battery_Dis_Voltage = Battery[battery_name]['Dis_Voltage']
    battery_Dis_Time = Battery[battery_name]['Dis_Time']
    
    # 将每个电池的数据添加到DataFrame中
    capacity_data = pd.Series(index=range(1, max_cycles + 1), dtype='float64')
    current_data = pd.Series(index=range(1, max_cycles + 1), dtype='object')  # 改为object类型
    voltage_data = pd.Series(index=range(1, max_cycles + 1), dtype='object')  # 改为object类型
    time_data = pd.Series(index=range(1, max_cycles + 1), dtype='object')     # 改为object类型
    
    capacity_data.loc[battery_cycles] = battery_capacity
    current_data.loc[battery_cycles] = battery_Dis_Current
    voltage_data.loc[battery_cycles] = battery_Dis_Voltage
    time_data.loc[battery_cycles] = battery_Dis_Time
    
    all_battery_data[f'{battery_name}_capacity'] = capacity_data
    all_battery_data[f'{battery_name}_Dis_Current'] = current_data
    all_battery_data[f'{battery_name}_Dis_Voltage'] = voltage_data
    all_battery_data[f'{battery_name}_Dis_Time'] = time_data

# 将数据保存到Excel文件
all_battery_data.to_excel(r'BatteryDataset/CALCE_BatteryDischargeData.xlsx', index=False)
print('数据已成功保存到 CALCE_BatteryCapacityData.xlsx')