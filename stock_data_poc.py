from alphagen_qlib.stock_data import StockData, FeatureType
import torch

def test_stock_data():
    # 1. 测试QLib初始化和不同时间范围
    print("=== 测试不同时间范围 ===")
    time_ranges = [
        ("2020-01-01", "2020-01-10"),  # 10天
        ("2020-01-01", "2020-01-31"),  # 1个月
        ("2020-01-01", "2020-12-31")   # 1年
    ]
    
    for start, end in time_ranges:
        print(f"\n时间范围: {start} 到 {end}")
        data = StockData("csi500", start, end, qlib_path="./data/qlib_data/cn_data_rolling")
        print(f"加载天数: {data.n_days}")
        print(f"特征维度: {data.n_features}")
        print(f"股票数量: {data.n_stocks}")
        print(f"数据形状: {data.data.shape}")
    
    # 2. 测试raw模式
    print("\n=== 测试raw模式 ===")
    features = [FeatureType.OPEN, FeatureType.CLOSE, FeatureType.VOLUME]
    data_raw = StockData("csi500", "2020-01-01", "2020-01-10", 
                        features=features, raw=True, qlib_path="your_qlib_data_path")
    print("Raw模式数据样本:")
    print(data_raw.data[:5])  # 打印前5天数据
    
    # 3. 测试特征维度
    print("\n=== 测试特征维度 ===")
    feature_combinations = [
        [FeatureType.OPEN],  # 单特征
        [FeatureType.OPEN, FeatureType.CLOSE],  # 双特征
        list(FeatureType)  # 全部特征
    ]
    
    for features in feature_combinations:
        print(f"\n特征组合: {[f.name for f in features]}")
        data = StockData("csi500", "2020-01-01", "2020-01-10", 
                        features=features, qlib_path="your_qlib_data_path")
        print(f"实际特征数: {data.n_features} (预期: {len(features)})")
    
    # 4. 测试CUDA支持
    print("\n=== 测试CUDA支持 ===")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("CUDA设备可用")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用CPU")
    
    data_cuda = StockData("csi500", "2020-01-01", "2020-01-10", 
                         device=device, qlib_path="your_qlib_data_path")
    print(f"数据设备: {data_cuda.data.device}")
    print(f"是否在CUDA上: {data_cuda.data.is_cuda}")

if __name__ == "__main__":
    test_stock_data()