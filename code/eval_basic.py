"""
基础评估脚本 - 适用于只有 generated 和 expected 字段的结果文件
不需要 type, ia_status, grounding 等额外字段
"""
import json
import re
import os
import sys
from collections import Counter
from datetime import datetime

def extract_code_content(text):
    """从模型输出中提取代码内容"""
    if not text or not text.strip():
        return ""
    
    # 匹配花括号 {...}
    brace_matches = re.findall(r'\{(.*?)\}', text, re.DOTALL)
    if brace_matches:
        return "\n".join(brace_matches)
    
    return text.strip()

def normalize_instruction(inst):
    """标准化单条指令"""
    inst = re.sub(r'\((\w+):(\w+)\)', r'(\2)', inst)
    inst = inst.rstrip(';')
    inst = inst.replace(" ", "")
    inst = inst.replace("\n", "")
    return inst

def compute_accuracy(generated_texts, expected_texts):
    """计算准确率指标"""
    print(f"样本数量: {len(generated_texts)}")
    correct_num = 0
    tp = 0
    all_pre = 0
    all_gold = 0
    
    for generated_text, expected_text in zip(generated_texts, expected_texts):
        # 处理生成的文本
        extracted_code = extract_code_content(generated_text)
        extracted_code = extracted_code.replace("\n", ",").replace(";", ",").replace(" ", "")
        generated_list = [x.strip() for x in extracted_code.split(",") if x.strip()]
        
        # 处理期望的文本
        expected_text = expected_text.replace("'''", "")
        expected_text = expected_text.replace("\n", ",").replace(";", ",").replace(" ", "")
        expected_list = [x.strip() for x in expected_text.split(",") if x.strip()]
        
        # 标准化处理
        generated_list = [normalize_instruction(x) for x in generated_list]
        expected_list = [normalize_instruction(x) for x in expected_list]
        
        # 使用 Counter 进行比较
        generated_counter = Counter(generated_list)
        expected_counter = Counter(expected_list)
        
        if generated_counter == expected_counter:
            correct_num += 1
        
        # 计算 TP
        intersection = generated_counter & expected_counter
        tp += len(list(intersection.elements()))
        all_pre += len(generated_list)
        all_gold += len(expected_list)
    
    # 计算指标
    em = correct_num / len(generated_texts) if generated_texts else 0
    precision = tp / all_pre if all_pre > 0 else 0
    recall = tp / all_gold if all_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  EM (Exact Match): {em:.4f} ({correct_num}/{len(generated_texts)})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    return em, precision, recall, f1

def main(result_file, log_dir="logs"):
    """主评估函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(current_dir, log_dir)
    os.makedirs(log_path, exist_ok=True)
    
    result_basename = os.path.splitext(os.path.basename(result_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("HomeBench 基础评估报告")
    print("=" * 60)
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果文件: {result_file}")
    print("=" * 60)
    
    # 读取结果文件
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception as e:
                    print(f"警告: 跳过无效行: {e}")
                    continue
    
    print(f"\n总样本数: {len(results)}")
    
    # 提取数据
    generated_texts = [r.get("generated", "") for r in results]
    expected_texts = [r.get("expected", "") for r in results]
    
    # 计算指标
    print("\n【整体性能】")
    em, precision, recall, f1 = compute_accuracy(generated_texts, expected_texts)
    
    # 保存结果
    summary = {
        "timestamp": timestamp,
        "result_file": result_file,
        "total_samples": len(results),
        "metrics": {
            "em": em,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    }
    
    # 保存JSON到logs目录
    summary_file = os.path.join(log_path, f"eval_{result_basename}_{timestamp}_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n汇总已保存至: {summary_file}")
    print("=" * 60)
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate basic test results")
    parser.add_argument("input", nargs="?", default="qwen_few_shot_5k_test_result.json", help="结果文件")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    
    # 查找文件
    result_file = os.path.join(results_dir, args.input)
    if not os.path.exists(result_file):
        result_file = os.path.join(current_dir, args.input)
    if not os.path.exists(result_file):
        result_file = args.input
    
    if os.path.exists(result_file):
        main(result_file, log_dir=args.log_dir)
    else:
        print(f"错误: 找不到文件 {args.input}")
        print(f"尝试的路径:")
        print(f"  1. {os.path.join(results_dir, args.input)}")
        print(f"  2. {os.path.join(current_dir, args.input)}")
        print(f"  3. {args.input}")
