"""
消融实验脚本 - 基于 model_test_ia.py
测试不同模块组合的性能：
1. Full Framework: IA + CodeGen + Grounding
2. No IA: CodeGen + Grounding
3. No Grounding: IA + CodeGen
4. Baseline: CodeGen Only
"""
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import time
import random
import argparse

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认使用完整测试集，可通过命令行参数指定子集
TEST_DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'test_data.jsonl')
HOME_STATUS_PATH = os.path.join(BASE_DIR, "dataset", "home_status_method.jsonl")
SYSTEM_PATH = os.path.join(BASE_DIR, "code", "system.txt")
EXAMPLE_PATH = os.path.join(BASE_DIR, "code", "example.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-7B-Instruct")
RESULTS_DIR = os.path.join(BASE_DIR, "code", "results")

# ============================================
# 意图分析 Prompt（优化版 - 增加置信度）
# ============================================
IA_SYSTEM = """You are a smart home intent analyzer. Your task is to verify if each operation in the user command can be executed.

CRITICAL VALIDATION RULES:
1. Room Existence: Check if the mentioned room exists in <home_state>
2. Device Existence: Check if the device exists in that specific room
3. Attribute Support: Check if the device supports the requested attribute/action
4. Attribute Values: Verify the value is within valid range or options

CONFIDENCE SCORING:
- confidence: 0.95-1.0 if you are CERTAIN about the validation result (room/device clearly exists or doesn't exist)
- confidence: 0.7-0.9 if you need to infer or the information is ambiguous
- confidence: 0.0-0.7 if you are uncertain

IMPORTANT:
- If a room doesn't exist, ALL operations for that room are INVALID with HIGH confidence
- If a device doesn't exist in the specified room, the operation is INVALID with HIGH confidence
- If a device doesn't support an attribute (not listed in its attributes), the operation is INVALID with HIGH confidence
- Check EACH operation independently and assign confidence to each

Output ONLY valid JSON in this exact format:
{"operations": [{"desc": "operation description", "valid": true/false, "reason": "specific reason", "confidence": 0.0-1.0, "order": 0}], "all_valid": true/false}"""

# Few-shot 示例（优化版：更清晰的验证逻辑）
IA_FEW_SHOT_EXAMPLES = """
# Valid Example 1: Single operation, device exists
<home_state>
living_room:
  light
    state: on
    brightness: 80 (range: 0 - 100)
</home_state>
<command>Turn off the light in living room</command>
Output: {"operations": [{"desc": "turn off light in living_room", "valid": true, "reason": "living_room exists, light exists, supports turn_off"}], "all_valid": true}

# Valid Example 2: Multiple operations, all devices exist
<home_state>
bedroom:
  light
    state: on
    brightness: 50
  curtain
    state: open
    degree: 100
</home_state>
<command>Close the curtain and turn on light in bedroom</command>
Output: {"operations": [{"desc": "close curtain in bedroom", "valid": true, "reason": "bedroom exists, curtain exists, supports close"}, {"desc": "turn on light in bedroom", "valid": true, "reason": "bedroom exists, light exists, supports turn_on"}], "all_valid": true}

# Invalid Example 1: Room doesn't exist
<home_state>
living_room:
  light
    state: on
</home_state>
<command>Turn on heater in bedroom</command>
Output: {"operations": [{"desc": "turn on heater in bedroom", "valid": false, "reason": "bedroom does not exist in home_state"}], "all_valid": false}

# Invalid Example 2: Multiple operations, all devices missing
<home_state>
kitchen:
  light
    state: off
</home_state>
<command>Turn on AC and heater in bedroom</command>
Output: {"operations": [{"desc": "turn on AC in bedroom", "valid": false, "reason": "bedroom does not exist in home_state"}, {"desc": "turn on heater in bedroom", "valid": false, "reason": "bedroom does not exist in home_state"}], "all_valid": false}

# Mixed Example 1: Some valid, some invalid
<home_state>
bedroom:
  light
    state: on
    brightness: 70
  air_conditioner
    state: off
    temperature: 26
</home_state>
<command>Turn on light and heater in bedroom</command>
Output: {"operations": [{"desc": "turn on light in bedroom", "valid": true, "reason": "bedroom exists, light exists, supports turn_on"}, {"desc": "turn on heater in bedroom", "valid": false, "reason": "heater does not exist in bedroom"}], "all_valid": false}

# Mixed Example 2: Attribute doesn't exist
<home_state>
living_room:
  light
    state: on
    brightness: 50
  curtain
    state: open
    degree: 80
</home_state>
<command>Close curtain, turn off light, and turn on humidifier in living room</command>
Output: {"operations": [{"desc": "close curtain in living_room", "valid": true, "reason": "living_room exists, curtain exists, supports close"}, {"desc": "turn off light in living_room", "valid": true, "reason": "living_room exists, light exists, supports turn_off"}, {"desc": "turn on humidifier in living_room", "valid": false, "reason": "humidifier does not exist in living_room"}], "all_valid": false}

"""

IA_USER = """<home_state>
{state}
</home_state>

<methods>
{methods}
</methods>

<command>
{command}
</command>

Analyze EACH operation carefully. Check room existence, device existence, and attribute support.
Output ONLY the JSON, no other text."""


# ============================================
# 数据格式转换
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
# Grounding 验证器（细粒度版本）
# ============================================
class GroundingValidator:
    """基于 home_status_method.jsonl 的 Grounding 验证"""
    
    def __init__(self, home_status, methods):
        self.rooms = set()
        self.devices = {}
        self.device_attrs = {}
        self.valid_methods = set()
        
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
        """验证单个方法调用"""
        call_str = call_str.strip()
        
        if call_str == "error_input":
            return True, "error_input"
        
        # 解析调用
        match = re.match(r'^([a-z_]+)\.([a-z_]+)\.([a-z_]+)\(.*\)$', call_str, re.IGNORECASE)
        if not match:
            match2 = re.match(r'^([a-z_]+)\.([a-z_]+)\(.*\)$', call_str, re.IGNORECASE)
            if match2:
                device, op = match2.groups()
                method_key = f"{device}.{op}"
                if method_key in self.valid_methods:
                    return True, "valid"
                return False, f"invalid_method:{method_key}"
            return False, f"invalid_format:{call_str}"
        
        room, device, op = match.groups()
        
        if room not in self.rooms:
            return False, f"room_not_exist:{room}"
        
        if room not in self.devices or device not in self.devices[room]:
            return False, f"device_not_exist:{room}.{device}"
        
        method_key = f"{room}.{device}.{op}"
        if method_key not in self.valid_methods:
            return False, f"method_not_exist:{method_key}"
        
        return True, "valid"
    
    def validate_output(self, output_str):
        """验证完整输出（细粒度版本）"""
        match = re.findall(r'\{([^}]*)\}', output_str)
        if not match:
            return False, [], ["no_code_block"]
        
        content = match[0].strip()
        if not content:
            return False, [], ["empty_code"]
        
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
# 选择性 Grounding 验证
# ============================================
def selective_grounding_validate(raw_output, ia_diagnosis, validator, confidence_threshold=0.9):
    """
    选择性 Grounding 验证：只验证低置信度的操作
    
    Args:
        raw_output: LLM 生成的原始输出
        ia_diagnosis: IA 模块的诊断结果（包含 operations 和 confidence）
        validator: GroundingValidator 实例
        confidence_threshold: 置信度阈值，高于此值跳过验证
    
    Returns:
        final_output: 验证后的最终输出
        stats: 验证统计信息
    """
    
    # 解析输出
    match = re.findall(r'\{([^}]*)\}', raw_output)
    if not match:
        return raw_output, {"error": "no_code_block", "high_confidence_skipped": 0, "low_confidence_validated": 0}
    
    content = match[0].strip()
    if not content:
        return "{error_input}", {"error": "empty_code", "high_confidence_skipped": 0, "low_confidence_validated": 0}
    
    calls = [c.strip() for c in content.split(",") if c.strip()]
    operations = ia_diagnosis.get("operations", [])
    
    # 统计信息
    stats = {
        "total_operations": len(calls),
        "high_confidence_skipped": 0,
        "low_confidence_validated": 0,
        "validation_errors": 0
    }
    
    # 如果操作数量不匹配，全部验证
    if len(calls) != len(operations):
        all_valid, validated_calls, reasons = validator.validate_output(raw_output)
        stats["error"] = "operation_count_mismatch"
        stats["low_confidence_validated"] = len(calls)
        return "{" + ",".join(validated_calls) + "}", stats
    
    # 按 order 排序（确保顺序正确）
    operations_sorted = sorted(operations, key=lambda x: x.get("order", 0))
    
    validated = []
    for i, (call, op) in enumerate(zip(calls, operations_sorted)):
        confidence = op.get("confidence", 0.0)
        
        if confidence >= confidence_threshold:
            # 高置信度：跳过 Grounding 验证，直接信任 IA
            validated.append(call)
            stats["high_confidence_skipped"] += 1
        else:
            # 低置信度：进行 Grounding 验证
            is_valid, reason = validator.validate_call(call)
            if is_valid:
                validated.append(call)
            else:
                validated.append("error_input")
                stats["validation_errors"] += 1
            stats["low_confidence_validated"] += 1
    
    final_output = "{" + ",".join(validated) + "}"
    return final_output, stats


# ============================================
# 数据集类
# ============================================
class AblationDataset:
    def __init__(self, tokenizer, use_few_shot=True, sample_size=None, seed=42, test_data_path=None):
        self.tokenizer = tokenizer
        
        # 使用指定的测试数据路径，如果没有指定则使用默认路径
        data_path = test_data_path if test_data_path else TEST_DATA_PATH
        
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if sample_size and sample_size < len(lines):
            random.seed(seed)
            lines = random.sample(lines, sample_size)
        
        with open(HOME_STATUS_PATH, "r", encoding="utf-8") as f:
            home_data = {}
            for line in f:
                d = json.loads(line)
                home_data[d["home_id"]] = {"home_status": d["home_status"], "method": d["method"]}
        
        with open(SYSTEM_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        
        examples = ""
        if use_few_shot and os.path.exists(EXAMPLE_PATH):
            with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
                examples = f.read()
        
        self.data = []
        for line in lines:
            case = json.loads(line)
            hid = case["home_id"]
            home_status = home_data[hid]["home_status"]
            methods = home_data[hid]["method"]
            
            state_str, method_str = chang_json2str(home_status, methods)
            
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
        
        # 性能统计
        self.total_tokens = 0
        self.total_calls = 0
        self.total_time = 0.0
    
    @torch.inference_mode()
    def generate(self, prompt, max_tokens=512, is_chat=True):
        """生成回复，并统计性能指标"""
        start_time = time.time()
        
        if is_chat:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 统计输入 tokens
        input_tokens = len(inputs['input_ids'][0])
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        # 统计输出 tokens
        output_tokens = len(outputs[0]) - input_tokens
        
        response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        # 更新统计
        elapsed_time = time.time() - start_time
        self.total_tokens += (input_tokens + output_tokens)
        self.total_calls += 1
        self.total_time += elapsed_time
        
        return response, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency": elapsed_time
        }
    
    def get_stats(self):
        """获取性能统计"""
        return {
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "avg_tokens_per_call": self.total_tokens / self.total_calls if self.total_calls > 0 else 0,
            "avg_latency": self.total_time / self.total_calls if self.total_calls > 0 else 0
        }
    
    def reset_stats(self):
        """重置统计"""
        self.total_tokens = 0
        self.total_calls = 0
        self.total_time = 0.0


# ============================================
# 意图分析
# ============================================
def oracle_intent_analysis(item):
    """Oracle意图分析：使用数据集真实标签（理论上限）"""
    item_type = item["type"]
    
    # 单指令类型
    if item_type == "normal":
        diagnosis = {"summary": f"Oracle: all_valid (type: {item_type})"}
        return "all_valid", {"oracle": True, "type": item_type}, diagnosis
    elif item_type in ["unexist_device", "unexist_attribute"]:
        diagnosis = {"summary": f"Oracle: all_invalid (type: {item_type})"}
        return "all_invalid", {"oracle": True, "type": item_type}, diagnosis
    
    # 多指令类型
    if item_type.startswith("multi"):
        if "_normal" in item_type:
            diagnosis = {"summary": f"Oracle: all_valid (type: {item_type})"}
            return "all_valid", {"oracle": True, "type": item_type}, diagnosis
        elif "_mix" in item_type:
            diagnosis = {"summary": f"Oracle: mixed (type: {item_type})"}
            return "mixed", {"oracle": True, "type": item_type}, diagnosis
        elif "_unexist_device" in item_type or "_unexist_attribute" in item_type:
            diagnosis = {"summary": f"Oracle: all_invalid (type: {item_type})"}
            return "all_invalid", {"oracle": True, "type": item_type}, diagnosis
    
    # 默认（未知类型）
    diagnosis = {"summary": f"Oracle: all_valid (type: {item_type}, unknown)"}
    return "all_valid", {"oracle": True, "type": item_type}, diagnosis


def intent_analysis(model, item, use_few_shot=True):
    """Stage 1: 意图分析（使用模型），返回诊断理由"""
    # 构建 prompt
    if use_few_shot:
        # 使用 few-shot 示例
        user_prompt = IA_FEW_SHOT_EXAMPLES + "\n# Now analyze this command:\n\n" + IA_USER.format(
            state=item["state_str"],
            methods=item["method_str"],
            command=item["input"]
        )
    else:
        # 不使用 few-shot
        user_prompt = IA_USER.format(
            state=item["state_str"],
            methods=item["method_str"],
            command=item["input"]
        )
    
    messages = [
        {"role": "system", "content": IA_SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response, perf_stats = model.generate(text, max_tokens=256, is_chat=False)
    
    # 构建诊断理由
    diagnosis = {
        "raw_response": response[:200],  # 保留前200字符
        "perf": perf_stats
    }
    
    try:
        # 更鲁棒的 JSON 提取
        if "{" in response and "}" in response:
            # 尝试提取 JSON
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]
            
            # 清理可能的格式问题
            json_str = json_str.strip()
            
            result = json.loads(json_str)
            
            # 验证必需字段
            if "operations" not in result or not isinstance(result["operations"], list):
                raise ValueError("Missing or invalid 'operations' field")
            
            all_valid = result.get("all_valid", True)
            operations = result.get("operations", [])
            
            # 确保每个操作都有必需字段（包括 confidence 和 order）
            for i, op in enumerate(operations):
                if "valid" not in op:
                    op["valid"] = True  # 默认为 valid
                if "desc" not in op:
                    op["desc"] = "unknown operation"
                if "reason" not in op:
                    op["reason"] = "no reason provided"
                if "confidence" not in op:
                    op["confidence"] = 0.5  # 默认中等置信度
                if "order" not in op:
                    op["order"] = i  # 默认按出现顺序
            
            valid_count = sum(1 for op in operations if op.get("valid", True))
            invalid_count = len(operations) - valid_count
            
            # 计算平均置信度
            avg_confidence = sum(op.get("confidence", 0.5) for op in operations) / len(operations) if operations else 0.0
            
            # 提取诊断理由
            diagnosis["operations"] = operations
            diagnosis["valid_count"] = valid_count
            diagnosis["invalid_count"] = invalid_count
            diagnosis["avg_confidence"] = avg_confidence
            
            # 生成简短的诊断文本
            if invalid_count == 0:
                diagnosis["summary"] = "All operations are valid and executable."
                return "all_valid", result, diagnosis
            elif valid_count == 0:
                diagnosis["summary"] = "All operations are invalid. " + "; ".join([op.get("reason", "") for op in operations[:3]])
                return "all_invalid", result, diagnosis
            else:
                valid_ops = [op.get("desc", "unknown") for op in operations if op.get("valid", True)]
                invalid_ops = [op.get("desc", "unknown") for op in operations if not op.get("valid", True)]
                diagnosis["summary"] = f"Mixed: {valid_count} valid, {invalid_count} invalid. Valid: {', '.join(valid_ops[:2])}. Invalid: {', '.join(invalid_ops[:2])}."
                return "mixed", result, diagnosis
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        diagnosis["error"] = str(e)
        diagnosis["summary"] = "Failed to parse IA response."
        diagnosis["parse_error_detail"] = f"Error: {str(e)}, Response: {response[:100]}"
    
    # 解析失败时，默认返回 all_valid（保守策略）
    return "all_valid", {"raw": response}, diagnosis


# ============================================
# 代码生成
# ============================================
def generate_code(model, item, ia_diagnosis=None):
    """Stage 2: 代码生成，可选地使用 IA 诊断理由"""
    
    # 如果有 IA 诊断，添加到 prompt 中
    if ia_diagnosis and ia_diagnosis.get("summary"):
        # 在原始 prompt 前添加 IA 诊断信息
        ia_hint = f"\n<intent_analysis>\nThe intent analyzer suggests: {ia_diagnosis['summary']}\n</intent_analysis>\n\n"
        enhanced_input = item["gen_input"].replace(
            "<User instructions:>",
            ia_hint + "<User instructions:>"
        )
        response, perf_stats = model.generate(enhanced_input, max_tokens=512, is_chat=True)
    else:
        response, perf_stats = model.generate(item["gen_input"], max_tokens=512, is_chat=True)
    
    return response, perf_stats


# ============================================
# 消融实验主函数
# ============================================
def run_ablation_study(
    experiment_name,
    use_ia=True,
    use_oracle=False,  # 新增：是否使用Oracle IA（真实标签）
    use_grounding=True,
    sample_size=None,
    seed=42,
    use_few_shot=True,
    test_data_path=None,  # 新增：自定义测试数据路径
    confidence_threshold=0.9  # 新增：选择性 Grounding 的置信度阈值
):
    """
    运行消融实验
    
    Args:
        experiment_name: 实验名称（用于输出文件名）
        use_ia: 是否使用意图分析模块
        use_oracle: 是否使用Oracle IA（真实标签，理论上限）
        use_grounding: 是否使用Grounding验证模块
        sample_size: 抽样数量
        seed: 随机种子
        use_few_shot: 是否使用few-shot
        test_data_path: 自定义测试数据路径（可选）
        confidence_threshold: 选择性 Grounding 的置信度阈值（默认 0.9）
    """
    print(f"\n{'='*70}")
    print(f"消融实验: {experiment_name}")
    print(f"{'='*70}")
    print(f"配置:")
    print(f"  - 意图分析 (IA): {'✓' if use_ia else '✗'}")
    if use_ia:
        print(f"  - Oracle模式: {'✓ (使用真实标签)' if use_oracle else '✗ (使用模型判断)'}")
    print(f"  - Grounding验证: {'✓' if use_grounding else '✗'}")
    print(f"  - Few-shot: {'✓' if use_few_shot else '✗'}")
    print(f"  - 样本数: {sample_size if sample_size else '全部'}")
    print(f"  - 随机种子: {seed}")
    print(f"{'='*70}\n")
    
    model = QwenModel()
    dataset = AblationDataset(
        model.tokenizer,
        use_few_shot=use_few_shot,
        sample_size=sample_size,
        seed=seed,
        test_data_path=test_data_path
    )
    
    results = []
    stats = {
        "total": 0,
        "ia_all_valid": 0,
        "ia_all_invalid": 0,
        "ia_mixed": 0,
        "grounding_valid": 0,
        "grounding_invalid": 0,
        "grounding_skipped": 0
    }
    
    # 性能统计
    perf_stats = {
        "ia_tokens": 0,
        "ia_calls": 0,
        "ia_time": 0.0,
        "codegen_tokens": 0,
        "codegen_calls": 0,
        "codegen_time": 0.0
    }
    
    start_time = time.time()
    
    for i in tqdm(range(len(dataset)), desc=f"Testing {experiment_name}"):
        item = dataset[i]
        stats["total"] += 1
        
        validator = GroundingValidator(item["home_status"], item["methods"])
        
        ia_status = "disabled"
        ia_diagnosis = None
        grounding_status = "disabled"
        
        # ===== Stage 1: 意图分析 (可选) =====
        if use_ia:
            if use_oracle:
                # 使用Oracle IA（真实标签）
                ia_status, ia_result, ia_diagnosis = oracle_intent_analysis(item)
            else:
                # 使用模型IA（支持 few-shot），返回诊断理由
                ia_status, ia_result, ia_diagnosis = intent_analysis(model, item, use_few_shot=use_few_shot)
                
                # 统计 IA 性能
                if "perf" in ia_diagnosis:
                    perf_stats["ia_tokens"] += ia_diagnosis["perf"]["total_tokens"]
                    perf_stats["ia_calls"] += 1
                    perf_stats["ia_time"] += ia_diagnosis["perf"]["latency"]
            
            if ia_status == "all_valid":
                stats["ia_all_valid"] += 1
            elif ia_status == "all_invalid":
                stats["ia_all_invalid"] += 1
            else:
                stats["ia_mixed"] += 1
        
        # ===== Stage 2: 代码生成 =====
        if use_ia and ia_status == "all_invalid":
            # 根据操作数量生成对应数量的 error_input
            operation_count = len(ia_diagnosis.get("operations", []))
            if operation_count > 1:
                final_output = "{" + ",".join(["error_input"] * operation_count) + "}"
            else:
                final_output = "{error_input}"
            grounding_status = "skipped"
            stats["grounding_skipped"] += 1
        else:
            # 传递 IA 诊断理由给代码生成
            raw_output, codegen_perf = generate_code(model, item, ia_diagnosis=ia_diagnosis if use_ia else None)
            
            # 统计 CodeGen 性能
            perf_stats["codegen_tokens"] += codegen_perf["total_tokens"]
            perf_stats["codegen_calls"] += 1
            perf_stats["codegen_time"] += codegen_perf["latency"]
            
            # ===== Stage 3: Grounding 验证 (可选) =====
            if use_grounding:
                # 选择性 Grounding：如果有 IA 且置信度阈值 > 0，使用选择性验证
                if use_ia and ia_diagnosis and confidence_threshold > 0:
                    final_output, grounding_stats = selective_grounding_validate(
                        raw_output, ia_diagnosis, validator, confidence_threshold
                    )
                    # 根据统计判断状态
                    if grounding_stats.get("high_confidence_skipped", 0) > 0:
                        grounding_status = "selective"
                    else:
                        grounding_status = "full_validated"
                else:
                    # 原有逻辑：全部验证
                    all_valid, validated_calls, reasons = validator.validate_output(raw_output)
                    
                    if all_valid:
                        stats["grounding_valid"] += 1
                        grounding_status = "valid"
                        final_output = raw_output
                    else:
                        stats["grounding_invalid"] += 1
                        grounding_status = "invalid"
                        # 细粒度验证：保留有效操作
                        final_output = "{" + ",".join(validated_calls) + "}"
            else:
                final_output = raw_output
                grounding_status = "disabled"
        
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
            "ia_diagnosis": ia_diagnosis.get("summary", "") if ia_diagnosis else "",
            "grounding": grounding_status
        })
        
        if (i + 1) % 100 == 0:
            save_results(results, f"ablation_{experiment_name}.json")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    save_results(results, f"ablation_{experiment_name}.json")
    
    # 输出统计
    print(f"\n{'='*70}")
    print(f"实验完成: {experiment_name}")
    print(f"{'='*70}")
    print(f"总耗时: {total_time:.1f}s")
    print(f"总样本数: {stats['total']}")
    print(f"平均耗时: {total_time/stats['total']:.2f}s/sample")
    
    # 性能统计
    print(f"\n【性能统计】")
    if use_ia and not use_oracle:
        print(f"  IA 模块:")
        print(f"    - 调用次数: {perf_stats['ia_calls']}")
        print(f"    - 总 Tokens: {perf_stats['ia_tokens']:,}")
        print(f"    - 平均 Tokens/调用: {perf_stats['ia_tokens']/perf_stats['ia_calls']:.0f}" if perf_stats['ia_calls'] > 0 else "    - 平均 Tokens/调用: 0")
        print(f"    - 总耗时: {perf_stats['ia_time']:.1f}s")
        print(f"    - 平均延迟: {perf_stats['ia_time']/perf_stats['ia_calls']:.2f}s/调用" if perf_stats['ia_calls'] > 0 else "    - 平均延迟: 0s/调用")
    
    print(f"  CodeGen 模块:")
    print(f"    - 调用次数: {perf_stats['codegen_calls']}")
    print(f"    - 总 Tokens: {perf_stats['codegen_tokens']:,}")
    print(f"    - 平均 Tokens/调用: {perf_stats['codegen_tokens']/perf_stats['codegen_calls']:.0f}" if perf_stats['codegen_calls'] > 0 else "    - 平均 Tokens/调用: 0")
    print(f"    - 总耗时: {perf_stats['codegen_time']:.1f}s")
    print(f"    - 平均延迟: {perf_stats['codegen_time']/perf_stats['codegen_calls']:.2f}s/调用" if perf_stats['codegen_calls'] > 0 else "    - 平均延迟: 0s/调用")
    
    total_tokens = perf_stats['ia_tokens'] + perf_stats['codegen_tokens']
    total_calls = perf_stats['ia_calls'] + perf_stats['codegen_calls']
    print(f"  总计:")
    print(f"    - 总 API 调用: {total_calls}")
    print(f"    - 总 Tokens: {total_tokens:,}")
    print(f"    - 平均 Tokens/sample: {total_tokens/stats['total']:.0f}")
    
    if use_ia:
        print(f"\n【意图分析统计】")
        if use_oracle:
            print(f"  模式: Oracle (使用真实标签)")
        else:
            print(f"  模式: Model (使用模型判断)")
        print(f"  All Valid: {stats['ia_all_valid']} ({stats['ia_all_valid']/stats['total']*100:.1f}%)")
        print(f"  All Invalid: {stats['ia_all_invalid']} ({stats['ia_all_invalid']/stats['total']*100:.1f}%)")
        print(f"  Mixed: {stats['ia_mixed']} ({stats['ia_mixed']/stats['total']*100:.1f}%)")
    
    if use_grounding:
        print(f"\n【Grounding验证统计】")
        print(f"  Valid: {stats['grounding_valid']} ({stats['grounding_valid']/stats['total']*100:.1f}%)")
        print(f"  Invalid (已修正): {stats['grounding_invalid']} ({stats['grounding_invalid']/stats['total']*100:.1f}%)")
        print(f"  Skipped: {stats['grounding_skipped']} ({stats['grounding_skipped']/stats['total']*100:.1f}%)")
    
    output_path = os.path.join(RESULTS_DIR, f"ablation_{experiment_name}.json")
    print(f"\n结果已保存: {output_path}")
    print(f"{'='*70}\n")
    
    # 保存性能统计
    perf_stats_file = os.path.join(RESULTS_DIR, f"ablation_{experiment_name}_perf.json")
    with open(perf_stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": experiment_name,
            "total_time": total_time,
            "total_samples": stats['total'],
            "avg_time_per_sample": total_time / stats['total'],
            "performance": perf_stats,
            "statistics": stats
        }, f, ensure_ascii=False, indent=2)
    print(f"性能统计已保存: {perf_stats_file}\n")
    
    return results, stats, perf_stats


def save_results(results, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验 - 测试不同模块组合")
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["full_framework", "oracle_ia", "no_ia", "no_grounding", "baseline"],
                       help="实验类型")
    parser.add_argument("--sample_size", type=int, default=None, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_few_shot", action="store_true", help="不使用 few-shot")
    parser.add_argument("--test_data", type=str, default=None, help="自定义测试数据路径（可选）")
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                       help="选择性 Grounding 的置信度阈值 (0.0-1.0)，默认 0.9")
    
    args = parser.parse_args()
    
    # 根据实验类型设置参数
    configs = {
        "full_framework": {
            "name": "full_framework",
            "use_ia": True,
            "use_oracle": False,
            "use_grounding": True,
            "description": "完整框架 (Model IA + CodeGen + Grounding)"
        },
        "oracle_ia": {
            "name": "oracle_ia",
            "use_ia": True,
            "use_oracle": True,
            "use_grounding": True,
            "description": "Oracle IA (真实标签 + CodeGen + Grounding)"
        },
        "no_ia": {
            "name": "no_ia",
            "use_ia": False,
            "use_oracle": False,
            "use_grounding": True,
            "description": "无意图分析 (CodeGen + Grounding)"
        },
        "no_grounding": {
            "name": "no_grounding",
            "use_ia": True,
            "use_oracle": False,
            "use_grounding": False,
            "description": "无Grounding验证 (Model IA + CodeGen)"
        },
        "baseline": {
            "name": "baseline",
            "use_ia": False,
            "use_oracle": False,
            "use_grounding": False,
            "description": "基线 (仅CodeGen)"
        }
    }
    
    config = configs[args.experiment]
    
    print(f"\n{'='*70}")
    print(f"开始消融实验")
    print(f"{'='*70}")
    print(f"实验: {config['description']}")
    if args.test_data:
        print(f"测试数据: {args.test_data}")
    print(f"{'='*70}\n")
    
    run_ablation_study(
        experiment_name=config["name"],
        use_ia=config["use_ia"],
        use_oracle=config["use_oracle"],
        use_grounding=config["use_grounding"],
        sample_size=args.sample_size,
        seed=args.seed,
        use_few_shot=not args.no_few_shot,
        test_data_path=args.test_data,
        confidence_threshold=args.confidence_threshold
    )
