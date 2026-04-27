import os
import re
import pandas as pd

def extract_iterations(content):
    """从文件内容中提取迭代步数"""
    # 尝试匹配pyscf格式的收敛信息
    pyscf_match = re.search(r'converged (\d+) \d+', content)
    if pyscf_match:
        return int(pyscf_match.group(1)) + 1
    
    # 尝试匹配tgd格式的迭代信息
    tgd_matches = re.findall(r'Step \d+:', content)
    if tgd_matches:
        return len(tgd_matches)
    
    # 尝试匹配davidson行
    davidson_matches = re.findall(r'davidson \d+ \d+', content)
    if davidson_matches:
        return len(davidson_matches)
    
    # 默认值
    return 13

def extract_l3_data(file_path, program_type):
    """提取L3测试数据"""
    data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    filename = os.path.basename(file_path)
    thread_match = re.search(r'(\d+)\.txt$', filename)
    if thread_match:
        data['T'] = int(thread_match.group(1))
    
    # 提取迭代步数
    iterations = extract_iterations(content)
    
    # 提取Wall_time_avg
    wall_time_match = re.search(r'Runtime \(RDTSC\) \[s\] STAT\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|', content)
    if wall_time_match:
        wall_time = float(wall_time_match.group(1))
        data['WT'] = wall_time / iterations if iterations > 0 else wall_time
    
    # 提取CPU_time_avg
    cpu_time_match = re.search(r'Runtime unhalted \[s\] STAT\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|', content)
    if cpu_time_match:
        cpu_time = float(cpu_time_match.group(1))
        data['CT'] = cpu_time / iterations if iterations > 0 else cpu_time
    
    # 提取CPI_avg
    cpi_match = re.search(r'CPI STAT\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|', content)
    if cpi_match:
        data['CPI'] = float(cpi_match.group(1))
    
    # 提取L3_bandwidth_avg
    l3_bw_match = re.search(r'L3 bandwidth \[MBytes/s\] STAT\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|', content)
    if l3_bw_match:
        data['L3_BW'] = float(l3_bw_match.group(1)) / 1024  # MB/s -> GB/s
    
    # 提取L2_PF_HIT_IN_L3_avg
    l2_hit_match = re.search(r'L2_PF_HIT_IN_L3 STAT\s+\|\s+PMC2\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.e+]+)\s+\|', content)
    if l2_hit_match:
        data['L2_HIT'] = float(l2_hit_match.group(1))
    
    # 提取L2_PF_MISS_IN_L3_avg
    l2_miss_match = re.search(r'L2_PF_MISS_IN_L3 STAT\s+\|\s+PMC3\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.e+]+)\s+\|', content)
    if l2_miss_match:
        data['L2_MISS'] = float(l2_miss_match.group(1))
    
    # 提取L2_CACHE_MISS_AFTER_L1_MISS_avg
    l2_cache_miss_match = re.search(r'L2_CACHE_MISS_AFTER_L1_MISS STAT\s+\|\s+PMC4\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.e+]+)\s+\|', content)
    if l2_cache_miss_match:
        data['L2_CM'] = float(l2_cache_miss_match.group(1))
    
    return data

def extract_mem_data(file_path, program_type):
    """提取MEM测试数据 - 提取总带宽"""
    data = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    filename = os.path.basename(file_path)
    thread_match = re.search(r'(\d+)\.txt$', filename)
    if thread_match:
        data['T'] = int(thread_match.group(1))
    
    # 提取Memory_bandwidth_avg (总带宽)
    mem_bw_match = re.search(r'Memory bandwidth \[MBytes/s\] STAT\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|', content)
    if mem_bw_match:
        data['MEM_BW'] = float(mem_bw_match.group(1)) / 1024  # MB/s -> GB/s
    
    return data

def format_single_value(value):
    """格式化单个数值, 保留4位小数"""
    if pd.isna(value):
        return value
    
    try:
        num = float(value)
        if abs(num) >= 1e5 or (0 < abs(num) < 1e-4):
            return f"{num:.4e}"
        else:
            return f"{num:.4f}"
    except:
        return value

def process_directory(base_dir, program, molecule, test_type):
    """处理特定目录的数据"""
    data_list = []
    
    # 构建目录路径
    dir_path = os.path.join(base_dir, program, molecule, test_type)
    
    if not os.path.exists(dir_path):
        print(f"目录不存在: {dir_path}")
        return []
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"目录中没有txt文件: {dir_path}")
        return []
    
    # 按线程数排序
    txt_files.sort(key=lambda x: int(re.search(r'(\d+)\.txt$', x).group(1)))
    
    for file in txt_files:
        print(f"处理文件: {dir_path}/{file}")
        file_path = os.path.join(dir_path, file)
        
        if test_type == 'l3':
            data = extract_l3_data(file_path, program)
        else:  # 'mem'
            data = extract_mem_data(file_path, program)
        
        if data:
            data_list.append(data)
    
    return data_list

def main():
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(CURR_PATH, 'benchmark')
    
    # 定义要处理的所有组合
    configurations = [
        {'program': 'pyscf', 'molecule': 'h2o_cc-pvdz', 'name': 'pyscf_h2o_ccpvdz'},
        {'program': 'pyscf', 'molecule': 'o2_6-31g', 'name': 'pyscf_o2_6-31g'},
        {'program': 'pyscf', 'molecule': 'hcn_6-31g', 'name': 'pyscf_hcn_6-31g'},
        {'program': 'tgd', 'molecule': 'h2o_cc-pvdz', 'name': 'tgd_h2o_ccpvdz'},
        {'program': 'tgd', 'molecule': 'o2_6-31g', 'name': 'tgd_o2_6-31g'},
        {'program': 'tgd', 'molecule': 'hcn_6-31g', 'name': 'tgd_hcn_6-31g'}
    ]
    
    all_results = {}
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"处理: {config['program']} - {config['molecule']}")
        print('='*60)
        
        # 处理L3数据
        l3_data = process_directory(benchmark_dir, config['program'], config['molecule'], 'l3')
        
        # 处理MEM数据
        mem_data = process_directory(benchmark_dir, config['program'], config['molecule'], 'mem')
        
        if l3_data and mem_data:
            # 创建DataFrame
            l3_df = pd.DataFrame(l3_data)
            mem_df = pd.DataFrame(mem_data)
            
            # 按线程数排序
            l3_df = l3_df.sort_values('T')
            mem_df = mem_df.sort_values('T')
            
            # 合并两个DataFrame
            merged_df = pd.merge(l3_df, mem_df, on='T', how='left')
            
            # 计算CPU利用率
            if 'CT' in merged_df.columns and 'WT' in merged_df.columns and 'T' in merged_df.columns:
                merged_df['CPU_UTIL'] = (merged_df['CT'] * merged_df['T']) / merged_df['WT']
            
            # 定义列顺序
            column_order = [
                'T',
                'WT',
                'CT',
                'CPU_UTIL',
                'CPI',
                'L3_BW',
                'L2_HIT',
                'L2_MISS',
                'L2_CM',
                'MEM_BW'
            ]
            
            # 只保留存在的列
            existing_columns = [col for col in column_order if col in merged_df.columns]
            merged_df = merged_df[existing_columns]
            
            # 格式化数据
            for col in merged_df.columns:
                if col != 'T':
                    merged_df[col] = merged_df[col].apply(lambda x: format_single_value(x))
            
            # 保存到字典中
            all_results[config['name']] = merged_df
            
            # 保存为单独的Excel文件
            output_file = f"{config['name']}_results.xlsx"
            merged_df.to_excel(output_file, index=False)
            print(f"✓ 数据已保存到: {output_file}")
            print(f"✓ 处理了 {len(l3_data)} 个L3文件和 {len(mem_data)} 个MEM文件")
            print("\n数据预览:")
            print(merged_df.to_string())
        else:
            print(f"⚠  {config['program']} - {config['molecule']} 数据不完整或不存在")
    
    # 创建汇总Excel文件 (所有结果在一个文件中, 不同sheet) 
    if all_results:
        with pd.ExcelWriter('all_benchmark_results.xlsx', engine='openpyxl') as writer:
            for name, df in all_results.items():
                df.to_excel(writer, sheet_name=name, index=False)
        
        print(f"\n{'='*60}")
        print("✓ 所有数据已汇总保存到: all_benchmark_results.xlsx")
        print("  每个配置保存在单独的sheet中")
        print('='*60)
        
        # 打印说明
        print("\n" + "="*80)
        print("列名说明:")
        print("="*80)
        print("T        : Threads - 线程数")
        print("WT       : Wall_time_per_iteration_s - 每次迭代的墙上时间, 来自 Runtime (RDTSC) [s] STAT 的 Avg 值除以迭代步数")
        print("CT       : CPU_time_per_thread_per_iteration_s - 每次迭代每个线程的CPU时间, 来自 Runtime unhalted [s] STAT 的 Avg 值除以迭代步数")
        print("CPU_UTIL : CPU_utilization - CPU利用率, 计算公式: (CT × T) / WT")
        print("CPI      : CPI_avg - 指令效率, 来自 CPI STAT 的 Avg 值")
        print("L3_BW    : L3_bandwidth_avg_GB/s - L3缓存带宽, 来自 L3 bandwidth [MBytes/s] STAT 的 Avg 值, 已转换为GB/s")
        print("L2_HIT   : L2_PF_HIT_IN_L3_avg - L2预取命中L3, 来自 L2_PF_HIT_IN_L3 STAT 的 Avg 值")
        print("L2_MISS  : L2_PF_MISS_IN_L3_avg - L2预取未命中L3, 来自 L2_PF_MISS_IN_L3 STAT 的 Avg 值")
        print("L2_CM    : L2_CACHE_MISS_AFTER_L1_MISS_avg - L1未命中后的L2缓存未命中, 来自 L2_CACHE_MISS_AFTER_L1_MISS STAT 的 Avg 值")
        print("MEM_BW   : Memory_bandwidth_avg_GB/s - 内存总带宽, 来自 Memory bandwidth [MBytes/s] STAT 的 Avg 值, 已转换为GB/s")
        print("\n" + "="*80)
        print("数据说明:")
        print("="*80)
        print("1. 所有数据均为平均值 (Avg) ")
        print("2. 时间数据 (WT和CT) 已除以迭代步数, 表示每次迭代的平均时间")
        print("3. 迭代步数提取规则:")
        print("   - pyscf: 从'converged X Y'提取")
        print("   - tgd: 从'Step XXX:'行数提取")
        print("   - 其他: 从'davidson X Y'行数提取")
        print("4. 带宽单位已从 MB/s 转换为 GB/s (除以1024) ")
        print("5. 所有数值保留4位小数")
        print("6. 科学计数法表示的数字也保留4位有效数字")
        print("7. 时间数据以L3测试为准")
    else:
        print("未找到有效数据")

if __name__ == "__main__":
    main()