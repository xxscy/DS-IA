"""
评估脚本 - 支持从 test_data.jsonl 获取类型信息
用于评估只有 generated 和 expected 字段的结果文件
"""
import json
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset", "test_data.jsonl")
LOGS_DIR = os.path.join(BASE_DIR, "code", "logs")


def load_test_data_types():
    """加载测试数据的类型信息"""
    type_map = {}
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            # 使用索引作为 key（因为结果文件是按顺序的）
            type_map[idx] = data.get("type", "unknown")
    return type_map


def normalize_code(code_str):
    """标准化代码字符串"""
    # 移除空格、换行、引号
    code_str = code_str.replace(" ", "").replace("\n", "").replace("'", "").replace('"', "")
    
    # 提取 {} 中的内容
    match = re.findall(r'\{([^}]*)\}', code_str)
    if match:
        code_str = match[0]
    
    # 分割成调用列表
    calls = [c.strip() for c in code_str.split(",") if c.strip()]
    return set(calls)


def calculate_metrics(generated, expected):
    """计算 EM, Precision, Recall, F1"""
    gen_set = normalize_code(generated)
    exp_set = normalize_code(expected)
    
    # EM (Exact Match)
    em = 1 if gen_set == exp_set else 0
    
    # Precision, Recall, F1
    if len(gen_set) == 0 and len(exp_set) == 0:
        precision = recall = f1 = 1.0
    elif len(gen_set) == 0:
        precision = recall = f1 = 0.0
    elif len(exp_set) == 0:
        precision = recall = f1 = 0.0
    else:
        intersection = len(gen_set & exp_set)
        precision = intersection / len(gen_set) if len(gen_set) > 0 else 0
        recall = intersection / len(exp_set) if len(exp_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return em, precision, recall, f1


def classify_type(type_str):
    """将详细类型分类为主要类型"""
    if type_str == "normal":
        return "VS"  # 有效单指令
    elif type_str in ["unexist_device", "unexist_attribute", "unexist_room"]:
        return "IS"  # 无效单指令
    elif type_str.startswith("multi") and "normal" in type_str:
        return "VM"  # 有效多指令
    elif type_str.startswith("multi") and ("unexist" in type_str or "invalid" in type_str):
        return "IM"  # 无效多指令
    elif type_str.startswith("multi") and "mix" in type_str:
        return "MM"  # 混合多指令
    else:
        return "unknown"


def evaluate(result_file, test_data_path=None):
    """评估结果文件"""
    print(f"\n{'='*60}")
    print(f"HomeBench 评估报告")
    print(f"{'='*60}")
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果文件: {result_file}")
    
    # 加载结果文件
    print(f"加载结果文件...")
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    
    print(f"总样本数: {len(results)}")
    
    # 检查结果文件是否包含 type 字段
    has_type_field = all('type' in r for r in results)
    
    if has_type_field:
        print(f"使用结果文件中的 type 字段")
        type_map = {idx: r.get('type', 'unknown') for idx, r in enumerate(results)}
    else:
        # 加载类型映射（从测试数据）
        print(f"从测试数据加载类型信息...")
        data_path = test_data_path if test_data_path else TEST_DATA_PATH
        type_map = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                type_map[idx] = data.get("type", "unknown")
    
    # 初始化统计
    stats = {
        "all": {"count": 0, "em": 0, "precision": [], "recall": [], "f1": []},
        "VS": {"count": 0, "em": 0, "precision": [], "recall": [], "f1": []},
        "IS": {"count": 0, "em": 0, "precision": [], "recall": [], "f1": []},
        "VM": {"count": 0, "em": 0, "precision": [], "recall": [], "f1": []},
        "IM": {"count": 0, "em": 0, "precision": [], "recall": [], "f1": []},
        "MM": {"count": 0, "em": 0, "precision": [], "recall": [], "f1": []},
    }
    
    # 评估每个样本
    for idx, result in enumerate(results):
        generated = result.get("generated", "")
        expected = result.get("expected", "")
        
        # 获取类型
        detail_type = type_map.get(idx, "unknown")
        main_type = classify_type(detail_type)
        
        # 计算指标
        em, precision, recall, f1 = calculate_metrics(generated, expected)
        
        # 更新统计
        stats["all"]["count"] += 1
        stats["all"]["em"] += em
        stats["all"]["precision"].append(precision)
        stats["all"]["recall"].append(recall)
        stats["all"]["f1"].append(f1)
        
        if main_type in stats:
            stats[main_type]["count"] += 1
            stats[main_type]["em"] += em
            stats[main_type]["precision"].append(precision)
            stats[main_type]["recall"].append(recall)
            stats[main_type]["f1"].append(f1)
    
    # 输出结果
    print(f"{'='*60}")
    print(f"分类评估结果")
    print(f"{'='*60}\n")
    
    type_names = {
        "all": "all (全部)",
        "VS": "VS (有效单指令)",
        "IS": "IS (无效单指令)",
        "VM": "VM (有效多指令)",
        "IM": "IM (无效多指令)",
        "MM": "MM (混合多指令)"
    }
    
    summary = {}
    
    for type_key in ["all", "VS", "IS", "VM", "IM", "MM"]:
        if stats[type_key]["count"] == 0:
            continue
        
        count = stats[type_key]["count"]
        em = stats[type_key]["em"] / count
        precision = sum(stats[type_key]["precision"]) / count
        recall = sum(stats[type_key]["recall"]) / count
        f1 = sum(stats[type_key]["f1"]) / count
        
        print(f"【{type_names[type_key]}】")
        print(f"样本数量: {count}")
        print(f"  EM (Exact Match): {em:.4f} ({stats[type_key]['em']}/{count})")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}\n")
        
        summary[type_key] = {
            "name": type_names[type_key],
            "count": count,
            "em": em,
            "em_count": stats[type_key]["em"],
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # 输出汇总表格
    print(f"{'='*60}")
    print(f"汇总表格")
    print(f"{'='*60}")
    print(f"{'类型':<20} {'数量':<10} {'EM':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print(f"{'-'*70}")
    for type_key in ["all", "VS", "IS", "VM", "IM", "MM"]:
        if type_key in summary:
            s = summary[type_key]
            print(f"{s['name']:<20} {s['count']:<10} {s['em']:<10.4f} {s['precision']:<10.4f} {s['recall']:<10.4f} {s['f1']:<10.4f}")
    
    # 保存日志
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = os.path.basename(result_file).replace(".json", "")
    log_file = os.path.join(LOGS_DIR, f"eval_{result_name}_{timestamp}.log")
    summary_file = os.path.join(LOGS_DIR, f"eval_{result_name}_{timestamp}_summary.json")
    
    # 保存详细日志
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"HomeBench 评估报告\n")
        f.write(f"{'='*60}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结果文件: {result_file}\n")
        f.write(f"总样本数: {len(results)}\n\n")
        
        for type_key in ["all", "VS", "IS", "VM", "IM", "MM"]:
            if type_key in summary:
                s = summary[type_key]
                f.write(f"【{s['name']}】\n")
                f.write(f"样本数量: {s['count']}\n")
                f.write(f"  EM: {s['em']:.4f} ({s['em_count']}/{s['count']})\n")
                f.write(f"  Precision: {s['precision']:.4f}\n")
                f.write(f"  Recall: {s['recall']:.4f}\n")
                f.write(f"  F1: {s['f1']:.4f}\n\n")
    
    # 保存 JSON 汇总
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"评估完成！")
    print(f"日志已保存至: {log_file}")
    print(f"汇总 JSON 已保存至: {summary_file}")
    print(f"{'='*60}")
    
    return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python eval_with_type.py <result_file> [test_data_path]")
        print("示例: python eval_with_type.py code/results/qwen_few_shot_5k_test_result.json")
        print("      python eval_with_type.py code/results/ablation_no_ia.json dataset/test_data_multi_types.jsonl")
        sys.exit(1)
    
    result_file = sys.argv[1]
    test_data_path = sys.argv[2] if len(sys.argv) >= 3 else None
    
    if not os.path.exists(result_file):
        print(f"错误: 文件不存在 - {result_file}")
        sys.exit(1)
    
    evaluate(result_file, test_data_path)
