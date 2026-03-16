"""
按照6种类别重新评估SAGE实验结果
分类：
- VS: 单一设备有效 (Valid Single)
- IS: 单一设备无效 (Invalid Single)
- VM: 多设备有效 (Valid Multi)
- IM: 多设备无效 (Invalid Multi)
- MM: 多设备混合 (Mixed Multi - 有效和无效混合)
- ALL: 所有指令
"""

import json
import argparse
from collections import defaultdict


def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_commands(command_str):
    """解析命令字符串，返回命令列表"""
    if not command_str:
        return []
    
    # 移除三引号
    command_str = command_str.strip().strip("'\"")
    
    # 如果是error_input，返回空列表
    if 'error_input' in command_str.lower():
        return ['error_input']
    
    # 分割命令
    commands = [cmd.strip() for cmd in command_str.split(',') if cmd.strip()]
    return commands


def categorize_test_case(case_type):
    """
    根据原始类型分类到6种类别
    
    分类规则：
    - VS (单一设备有效): normal
    - IS (单一设备无效): unexist_device, unexist_attribute
    - VM (多设备有效): multi2_normal, multi3_normal, multi4_normal, ...
    - IM (多设备无效): multi2_unexist_device, multi2_unexist_attribute, ...
    - MM (多设备混合): multi2_mix, multi3_mix, ..., multi10_mix
    """
    case_type = case_type.lower()
    
    # 单一设备有效
    if case_type == 'normal':
        return 'VS'
    
    # 单一设备无效
    if case_type in ['unexist_device', 'unexist_attribute']:
        return 'IS'
    
    # 多设备类型
    if case_type.startswith('multi'):
        # 多设备有效
        if 'normal' in case_type:
            return 'VM'
        # 多设备无效
        elif 'unexist' in case_type:
            return 'IM'
        # 多设备混合
        elif 'mix' in case_type:
            return 'MM'
    
    return 'UNKNOWN'


def calculate_metrics(ground_truth_cmds, predicted_cmds):
    """
    计算评估指标
    
    返回：
    - em: 精确匹配 (1 or 0)
    - precision: 精确率
    - recall: 召回率
    - f1: F1分数
    """
    # 精确匹配
    em = 1 if set(ground_truth_cmds) == set(predicted_cmds) else 0
    
    # 计算交集
    if not predicted_cmds:
        precision = 0.0
        recall = 0.0 if ground_truth_cmds else 1.0
    elif not ground_truth_cmds:
        precision = 0.0
        recall = 0.0
    else:
        correct = len(set(ground_truth_cmds) & set(predicted_cmds))
        precision = correct / len(predicted_cmds) if predicted_cmds else 0.0
        recall = correct / len(ground_truth_cmds) if ground_truth_cmds else 0.0
    
    # F1分数
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'em': em,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_by_category(results_file, test_data_file):
    """按6种类别评估结果"""
    
    # 加载数据
    results = load_jsonl(results_file)
    test_data = load_jsonl(test_data_file)
    
    # 创建test_data的查找字典
    test_lookup = {item['id']: item for item in test_data}
    
    # 按类别统计
    category_stats = {
        'VS': {'total': 0, 'correct': 0, 'precision': [], 'recall': [], 'f1': []},
        'IS': {'total': 0, 'correct': 0, 'precision': [], 'recall': [], 'f1': []},
        'VM': {'total': 0, 'correct': 0, 'precision': [], 'recall': [], 'f1': []},
        'IM': {'total': 0, 'correct': 0, 'precision': [], 'recall': [], 'f1': []},
        'MM': {'total': 0, 'correct': 0, 'precision': [], 'recall': [], 'f1': []},
        'ALL': {'total': 0, 'correct': 0, 'precision': [], 'recall': [], 'f1': []}
    }
    
    # 遍历结果
    for result in results:
        case_id = result.get('case_id')
        
        # 获取对应的测试数据
        test_case = test_lookup.get(case_id)
        if not test_case:
            continue
        
        # 获取类别
        case_type = result.get('ground_truth_type', test_case.get('type', ''))
        category = categorize_test_case(case_type)
        
        if category == 'UNKNOWN':
            continue
        
        # 解析命令
        ground_truth = parse_commands(result.get('ground_truth_output', ''))
        predicted = parse_commands(result.get('sage_output', ''))
        
        # 计算指标
        metrics = calculate_metrics(ground_truth, predicted)
        
        # 更新统计
        category_stats[category]['total'] += 1
        category_stats[category]['correct'] += metrics['em']
        category_stats[category]['precision'].append(metrics['precision'])
        category_stats[category]['recall'].append(metrics['recall'])
        category_stats[category]['f1'].append(metrics['f1'])
        
        # 更新ALL统计
        category_stats['ALL']['total'] += 1
        category_stats['ALL']['correct'] += metrics['em']
        category_stats['ALL']['precision'].append(metrics['precision'])
        category_stats['ALL']['recall'].append(metrics['recall'])
        category_stats['ALL']['f1'].append(metrics['f1'])
    
    # 计算平均值
    final_results = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            final_results[category] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'success_rate': stats['correct'] / stats['total'],
                'avg_precision': sum(stats['precision']) / len(stats['precision']),
                'avg_recall': sum(stats['recall']) / len(stats['recall']),
                'avg_f1': sum(stats['f1']) / len(stats['f1'])
            }
        else:
            final_results[category] = {
                'total': 0,
                'correct': 0,
                'success_rate': 0.0,
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_f1': 0.0
            }
    
    return final_results


def print_results(results):
    """打印结果表格"""
    print("\n" + "=" * 100)
    print("SAGE实验结果 - 按6种类别分析")
    print("=" * 100)
    print()
    
    # 表头
    print(f"{'类别':<10} {'说明':<25} {'样本数':<10} {'成功数':<10} {'成功率':<12} {'F1分数':<12}")
    print("-" * 100)
    
    # 类别说明
    category_names = {
        'VS': '单一设备有效',
        'IS': '单一设备无效',
        'VM': '多设备有效',
        'IM': '多设备无效',
        'MM': '多设备混合',
        'ALL': '所有指令'
    }
    
    # 按顺序打印
    order = ['VS', 'IS', 'VM', 'IM', 'MM', 'ALL']
    
    for category in order:
        if category in results:
            stats = results[category]
            name = category_names.get(category, category)
            print(f"{category:<10} {name:<25} {stats['total']:<10} {stats['correct']:<10} "
                  f"{stats['success_rate']*100:>10.2f}% {stats['avg_f1']*100:>10.2f}%")
    
    print("=" * 100)
    print()
    
    # 详细指标
    print("详细指标:")
    print("-" * 100)
    print(f"{'类别':<10} {'Precision':<15} {'Recall':<15} {'F1':<15}")
    print("-" * 100)
    
    for category in order:
        if category in results:
            stats = results[category]
            print(f"{category:<10} {stats['avg_precision']*100:>13.2f}% {stats['avg_recall']*100:>13.2f}% "
                  f"{stats['avg_f1']*100:>13.2f}%")
    
    print("=" * 100)


def save_results(results, output_file):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='按6种类别评估SAGE实验结果')
    parser.add_argument('--input', type=str, required=True,
                        help='实验结果文件路径 (experiment_results.jsonl)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据文件路径 (test_data.jsonl)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径 (可选)')
    
    args = parser.parse_args()
    
    # 评估
    results = evaluate_by_category(args.input, args.test_data)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if args.output:
        save_results(results, args.output)
    else:
        # 默认保存到输入文件同目录
        import os
        input_dir = os.path.dirname(args.input)
        output_file = os.path.join(input_dir, 'evaluation_by_category.json')
        save_results(results, output_file)


if __name__ == '__main__':
    main()
