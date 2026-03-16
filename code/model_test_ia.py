"""
基于 model_test.py 的意图分析增强版
Stage 1: Intent Analysis - 意图分析
Stage 2: Code Generation - 代码生成  
Gate: Grounding Validation - 基于环境信息验证
"""
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import random

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset", "test_data.jsonl")
HOME_STATUS_PATH = os.path.join(BASE_DIR, "dataset", "home_status_method.jsonl")
SYSTEM_PATH = os.path.join(BASE_DIR, "code", "system.txt")
EXAMPLE_PATH = os.path.join(BASE_DIR, "code", "example.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-7B-Instruct")
RESULTS_DIR = os.path.join(BASE_DIR, "code", "results")

# ============================================
# 意图分析 Prompt
# ============================================
IA_SYSTEM = """You are a smart home intent analyzer. Analyze if each operation in the user command is valid.
Check: 1) Does the room exist? 2) Does the device exist in that room? 3) Does the device support the action/attribute?
Output JSON: {"operations": [{"desc": "...", "valid": true/false, "reason": "..."}], "all_valid": true/false}"""

IA_USER = """<home_state>
{state}
</home_state>

<methods>
{methods}
</methods>

<command>
{command}
</command>

Analyze each operation and output JSON only."""


# ============================================
# 数据格式转换（保持原有格式）
# ============================================
def chang_json2str(state, methods):
    """转换状态和方法为字符串格式"""
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            state_str += "  state: " + state[room]["state"] + "\n"
            for attribute in state[room]["attributes"].keys():
                state_str += "  " + attribute + ": " + str(state[room]["attributes"][attribute]["value"])
                if "options" in state[room]["attributes"][attribute].keys():
                    state_str += " (options" + str(state[room]["attributes"][attribute]["options"]) + ")\n"
                elif "lowest" in state[room]["attributes"][attribute].keys():
                    state_str += " (range: " + str(state[room]["attributes"][attribute]["lowest"]) + " - " + str(state[room]["attributes"][attribute]["highest"]) + ")\n"
                else:
                    state_str += "\n"
        else:
            for device in state[room].keys():
                if device == "room_name":
                    continue
                state_str += "  " + device + "\n"
                state_str += "    state: " + state[room][device]["state"] + "\n"
                for attribute in state[room][device]["attributes"].keys():
                    state_str += "    " + attribute + ": " + str(state[room][device]["attributes"][attribute]["value"])
                    if "options" in state[room][device]["attributes"][attribute].keys():
                        state_str += " (options" + str(state[room][device]["attributes"][attribute]["options"]) + ")\n"
                    elif "lowest" in state[room][device]["attributes"][attribute].keys():
                        state_str += " (range: " + str(state[room][device]["attributes"][attribute]["lowest"]) + " - " + str(state[room][device]["attributes"][attribute]["highest"]) + ")\n"
                    else:
                        state_str += "\n"
    
    method_str = ""
    for method in methods:
        if method["room_name"] == "None":
            method_str += method["device_name"] + "." + method["operation"] + "("
        else:
            method_str += method["room_name"] + "." + method["device_name"] + "." + method["operation"] + "("
        if len(method["parameters"]) > 0:
            for parameter in method["parameters"]:
                method_str += parameter["name"] + ":" + parameter["type"] + ","
            method_str = method_str[:-1]
        method_str += ");\n"
    return state_str, method_str


# ============================================
# Grounding 验证器（基于环境信息）
# ============================================
class GroundingValidator:
    """基于 home_status_method.jsonl 的 Grounding 验证"""
    
    def __init__(self, home_status, methods):
        """
        home_status: 家庭设备状态 dict
        methods: 可用方法列表
        """
        self.rooms = set()
        self.devices = {}  # {room: set(devices)}
        self.device_attrs = {}  # {room.device: set(attributes)}
        self.valid_methods = set()  # {room.device.operation}
        
        # 解析设备状态
        for room, room_data in home_status.items():
            self.rooms.add(room)
            if room == "VacuumRobot":
                self.devices[room] = {"VacuumRobot"}
                self.device_attrs[f"{room}.VacuumRobot"] = set(room_data.get("attributes", {}).keys())
            else:
                self.devices[room] = set()
                for device, device_data in room_data.items():
                    if device == "room_name":
                        continue
                    self.devices[room].add(device)
                    attrs = set(device_data.get("attributes", {}).keys())
                    # 添加基本操作
                    attrs.add("state")
                    self.device_attrs[f"{room}.{device}"] = attrs
        
        # 解析可用方法
        for m in methods:
            room = m["room_name"]
            device = m["device_name"]
            op = m["operation"]
            if room == "None":
                self.valid_methods.add(f"{device}.{op}")
            else:
                self.valid_methods.add(f"{room}.{device}.{op}")
    
    def validate_call(self, call_str):
        """
        验证单个方法调用是否有效
        call_str: 如 "living_room.light.turn_on()" 或 "error_input"
        返回: (is_valid, reason)
        """
        call_str = call_str.strip()
        
        # error_input 是有效的
        if call_str == "error_input":
            return True, "error_input"
        
        # 解析调用
        match = re.match(r'^([a-z_]+)\.([a-z_]+)\.([a-z_]+)\(.*\)$', call_str, re.IGNORECASE)
        if not match:
            # 尝试匹配无房间的格式 device.operation()
            match2 = re.match(r'^([a-z_]+)\.([a-z_]+)\(.*\)$', call_str, re.IGNORECASE)
            if match2:
                device, op = match2.groups()
                method_key = f"{device}.{op}"
                if method_key in self.valid_methods:
                    return True, "valid"
                return False, f"invalid_method:{method_key}"
            return False, f"invalid_format:{call_str}"
        
        room, device, op = match.groups()
        
        # 检查房间
        if room not in self.rooms:
            return False, f"room_not_exist:{room}"
        
        # 检查设备
        if room not in self.devices or device not in self.devices[room]:
            return False, f"device_not_exist:{room}.{device}"
        
        # 检查方法
        method_key = f"{room}.{device}.{op}"
        if method_key not in self.valid_methods:
            return False, f"method_not_exist:{method_key}"
        
        return True, "valid"
    
    def validate_output(self, output_str):
        """
        验证完整输出
        output_str: 如 "{living_room.light.turn_on(),error_input}"
        返回: (all_valid, validated_calls, reasons)
        """
        # 提取 {} 中的内容
        match = re.findall(r'\{([^}]*)\}', output_str)
        if not match:
            return False, [], ["no_code_block"]
        
        content = match[0].strip()
        if not content:
            return False, [], ["empty_code"]
        
        # 分割调用
        calls = [c.strip() for c in content.split(",") if c.strip()]
        
        validated = []
        reasons = []
        all_valid = True
        
        for call in calls:
            is_valid, reason = self.validate_call(call)
            validated.append(call if is_valid else "error_input")
            reasons.append(reason)
            if not is_valid and call != "error_input":
                all_valid = False
        
        return all_valid, validated, reasons


# ============================================
# 数据集类（增强版）
# ============================================
class IAHomeAssistantDataset(Dataset):
    """带意图分析的数据集"""
    
    def __init__(self, tokenizer, use_few_shot=True, sample_size=None, seed=42):
        self.tokenizer = tokenizer
        
        # 读取测试数据
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 随机抽样
        if sample_size and sample_size < len(lines):
            random.seed(seed)
            lines = random.sample(lines, sample_size)
        
        # 读取环境状态
        with open(HOME_STATUS_PATH, "r", encoding="utf-8") as f:
            home_data = {}
            for line in f:
                d = json.loads(line)
                home_data[d["home_id"]] = {"home_status": d["home_status"], "method": d["method"]}
        
        # 读取 system prompt 和 examples
        with open(SYSTEM_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        examples = ""
        if use_few_shot and os.path.exists(EXAMPLE_PATH):
            with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
                examples = f.read()
        
        # 构建数据
        self.data = []
        for line in lines:
            case = json.loads(line)
            hid = case["home_id"]
            home_status = home_data[hid]["home_status"]
            methods = home_data[hid]["method"]
            
            state_str, method_str = chang_json2str(home_status, methods)
            
            # 构建代码生成 prompt（保持原格式）
            case_input = "-------------------------------\n" + \
                        "Here are the user instructions you need to reply to.\n" + \
                        "<User instructions:> \n" + case["input"] + "\n" + \
                        "<Machine instructions:>"
            home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:" + state_str + "\n</home_state>\n"
            device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:" + method_str + "\n</device_method>\n"
            
            gen_input = system_prompt + home_status_case + device_method_case
            if examples:
                gen_input += examples
            gen_input += case_input
            
            self.data.append({
                "id": case.get("id", len(self.data)),
                "input": case["input"],
                "output": case["output"],
                "type": case.get("type", "unknown"),
                "home_id": hid,
                "state_str": state_str,
                "method_str": method_str,
                "gen_input": gen_input,
                "home_status": home_status,
                "methods": methods
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================
# 模型类
# ============================================
class QwenModel:
    def __init__(self):
        print(f"Loading model: {MODEL_PATH}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        self.model.eval()
        print("Model loaded!")
    
    @torch.inference_mode()
    def generate(self, prompt, max_tokens=512, is_chat=True):
        """生成回复"""
        if is_chat:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        response = outputs[0][len(inputs['input_ids'][0]):]
        return self.tokenizer.decode(response, skip_special_tokens=True)


# ============================================
# 意图分析
# ============================================
def intent_analysis(model, item):
    """
    Stage 1: 意图分析
    返回: (status, analysis_result)
    status: "all_valid", "all_invalid", "mixed"
    """
    prompt = IA_USER.format(
        state=item["state_str"],
        methods=item["method_str"],
        command=item["input"]
    )
    
    messages = [
        {"role": "system", "content": IA_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = model.generate(text, max_tokens=256, is_chat=False)
    
    # 解析 JSON
    try:
        if "{" in response:
            json_str = response[response.find("{"):response.rfind("}")+1]
            result = json.loads(json_str)
            
            all_valid = result.get("all_valid", True)
            operations = result.get("operations", [])
            
            valid_count = sum(1 for op in operations if op.get("valid", True))
            invalid_count = len(operations) - valid_count
            
            if invalid_count == 0:
                return "all_valid", result
            elif valid_count == 0:
                return "all_invalid", result
            else:
                return "mixed", result
    except:
        pass
    
    # 解析失败，默认 all_valid
    return "all_valid", {"raw": response}


# ============================================
# 代码生成
# ============================================
def generate_code(model, item):
    """Stage 2: 代码生成"""
    response = model.generate(item["gen_input"], max_tokens=512, is_chat=True)
    return response


# ============================================
# 主测试函数
# ============================================
def run_test(
    sample_size=None,
    seed=42,
    use_few_shot=True,
    use_ia=True,
    output_file="qwen_ia_test_result.json"
):
    """
    运行测试
    use_ia: 是否启用意图分析
    """
    model = QwenModel()
    dataset = IAHomeAssistantDataset(
        model.tokenizer,
        use_few_shot=use_few_shot,
        sample_size=sample_size,
        seed=seed
    )
    
    results = []
    stats = {
        "total": 0,
        "ia_all_valid": 0,
        "ia_all_invalid": 0,
        "ia_mixed": 0,
        "grounding_valid": 0,
        "grounding_invalid": 0
    }
    
    print(f"\n{'='*60}")
    print(f"Test Config:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Seed: {seed}")
    print(f"  Few-shot: {use_few_shot}")
    print(f"  Intent Analysis: {use_ia}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for i in tqdm(range(len(dataset)), desc="Testing"):
        item = dataset[i]
        stats["total"] += 1
        
        # 创建 Grounding 验证器
        validator = GroundingValidator(item["home_status"], item["methods"])
        
        ia_status = "skipped"
        ia_result = None
        
        # ===== Stage 1: 意图分析 =====
        if use_ia:
            ia_status, ia_result = intent_analysis(model, item)
            
            if ia_status == "all_valid":
                stats["ia_all_valid"] += 1
            elif ia_status == "all_invalid":
                stats["ia_all_invalid"] += 1
            else:
                stats["ia_mixed"] += 1
        
        # ===== Stage 2: 代码生成 =====
        if use_ia and ia_status == "all_invalid":
            # 意图分析判断全部无效，直接返回 error_input
            final_output = "{error_input}"
            grounding_status = "skipped"
        else:
            # 生成代码
            raw_output = generate_code(model, item)
            
            # ===== Grounding 验证 =====
            all_valid, validated_calls, reasons = validator.validate_output(raw_output)
            
            if all_valid:
                stats["grounding_valid"] += 1
                grounding_status = "valid"
                final_output = raw_output
            else:
                stats["grounding_invalid"] += 1
                grounding_status = "invalid"
                # 用验证后的结果替换
                final_output = "{" + ",".join(validated_calls) + "}"
        
        # 提取最终代码
        match = re.findall(r'\{([^}]*)\}', final_output)
        if match:
            final_code = "{" + match[0] + "}"
        else:
            final_code = "{error_input}"
        
        results.append({
            "id": item["id"],
            "generated": final_code,
            "expected": item["output"],
            "type": item["type"],
            "ia_status": ia_status,
            "grounding": grounding_status
        })
        
        # 每 100 条保存一次
        if (i + 1) % 100 == 0:
            save_results(results, output_file)
            print(f"\nSaved {i+1} results")
    
    end_time = time.time()
    
    # 最终保存
    save_results(results, output_file)
    
    # 输出统计
    print(f"\n{'='*60}")
    print(f"Results (time: {end_time - start_time:.1f}s):")
    print(f"  Total: {stats['total']}")
    if use_ia:
        print(f"  IA All Valid: {stats['ia_all_valid']} ({stats['ia_all_valid']/stats['total']*100:.1f}%)")
        print(f"  IA All Invalid: {stats['ia_all_invalid']} ({stats['ia_all_invalid']/stats['total']*100:.1f}%)")
        print(f"  IA Mixed: {stats['ia_mixed']} ({stats['ia_mixed']/stats['total']*100:.1f}%)")
    print(f"  Grounding Valid: {stats['grounding_valid']} ({stats['grounding_valid']/stats['total']*100:.1f}%)")
    print(f"  Grounding Invalid: {stats['grounding_invalid']} ({stats['grounding_invalid']/stats['total']*100:.1f}%)")
    print(f"  Saved: {os.path.join(RESULTS_DIR, output_file)}")
    print(f"{'='*60}")
    
    return results, stats


def save_results(results, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================
# 入口
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=None, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, default="qwen_ia_test_result.json", help="输出文件名")
    parser.add_argument("--no_few_shot", action="store_true", help="不使用 few-shot")
    parser.add_argument("--no_ia", action="store_true", help="不使用意图分析")
    
    args = parser.parse_args()
    
    run_test(
        sample_size=args.sample_size,
        seed=args.seed,
        use_few_shot=not args.no_few_shot,
        use_ia=not args.no_ia,
        output_file=args.output
    )
