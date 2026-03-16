import json
import os
import random
from datetime import datetime
from tqdm import tqdm
import torch

# Import SAGE components
from sage_brain.coordinator import SAGECoordinator
from sage_brain.homebench_tool import QueryDevicesTool, ExecuteCommandTool

# Import LangChain and HuggingFace
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ================= Configuration =================
MODEL_PATH = "models/Qwen2.5-7B-Instruct"  # Local Qwen model path
DATA_PATH = "dataset/test_data.jsonl"
ENV_PATH = "dataset/home_status_method.jsonl"

# Create output directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results/qwen_experiment_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "experiment_results.jsonl")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.txt")
CONFIG_FILE = os.path.join(OUTPUT_DIR, "config.json")

TEST_LIMIT = 500  # Number of samples to test
RANDOM_SAMPLE = True  # Set to True for random sampling, False for first N samples
RANDOM_SEED = 42  # Set seed for reproducibility
# ===========================================


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_env_lookup(env_data):
    """Build home_id -> environment details lookup dictionary"""
    lookup = {}
    for item in env_data:
        h_id = item.get('home_id')
        if h_id is not None:
            lookup[h_id] = item
    return lookup


def load_local_qwen_model(model_path):
    """Load local Qwen model using HuggingFace"""
    print(f"📦 Loading Qwen model from: {model_path}")
    print("   This may take a few minutes...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model with GPU support if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1
    )
    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    
    print("✅ Qwen model loaded successfully!")
    return llm


def main():
    print("🚀 Starting HomeBench-SAGE Experiment with Qwen...")
    print("=" * 80)
    print("SAGE Workflow: User Query → Query Devices → Execute Command")
    print("=" * 80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # Save configuration
    config = {
        "model": "Qwen2.5-7B-Instruct",
        "model_path": MODEL_PATH,
        "timestamp": TIMESTAMP,
        "test_limit": TEST_LIMIT,
        "random_sample": RANDOM_SAMPLE,
        "random_seed": RANDOM_SEED if RANDOM_SAMPLE else None,
        "data_path": DATA_PATH,
        "env_path": ENV_PATH,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Configuration saved to: {CONFIG_FILE}")
    
    # 1. Load data
    print(f"\n📂 Loading data from {DATA_PATH}...")
    test_cases = load_jsonl(DATA_PATH)
    env_data = load_jsonl(ENV_PATH)
    env_lookup = build_env_lookup(env_data)
    print(f"✅ Loaded {len(test_cases)} test cases and {len(env_lookup)} environments.")

    # 2. Load Qwen model and initialize SAGE Brain
    print("\n🧠 Initializing SAGE Brain with Qwen model...")
    llm = load_local_qwen_model(MODEL_PATH)
    
    # Create all tools
    query_tool = QueryDevicesTool()
    execute_tool = ExecuteCommandTool()
    
    # Initialize Coordinator with tools
    sage = SAGECoordinator(llm=llm, tools=[query_tool, execute_tool])
    print("✅ SAGE Brain initialized with 2 tools:")
    print("   1. query_devices - Discover available devices")
    print("   2. execute_command - Execute smart home commands")

    # 3. Select test cases (random sampling or sequential)
    results = []
    
    if RANDOM_SAMPLE and TEST_LIMIT and TEST_LIMIT < len(test_cases):
        # Random sampling
        random.seed(RANDOM_SEED)
        target_cases = random.sample(test_cases, TEST_LIMIT)
        print(f"\n🎲 Randomly sampled {len(target_cases)} cases from {len(test_cases)} total (seed={RANDOM_SEED})")
    else:
        # Sequential selection
        target_cases = test_cases[:TEST_LIMIT] if TEST_LIMIT else test_cases
        print(f"\n📋 Selected first {len(target_cases)} cases from {len(test_cases)} total")
    
    print(f"\n🔄 Running execution on {len(target_cases)} samples...")
    print("=" * 80)
    
    # Clear output file if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    success_count = 0
    start_time = datetime.now()
    
    # Use tqdm for progress bar
    for idx, case in enumerate(tqdm(target_cases, desc="Processing", unit="case")):
        home_id = case.get('home_id')
        user_query = case.get('input')
        ground_truth_output = case.get('output', '')
        case_type = case.get('type', 'unknown')
        
        # Get environment
        env_item = env_lookup.get(home_id)
        if not env_item:
            continue
        
        # === Core: SAGE Execution ===
        try:
            result = sage.execute_homebench(
                user_query=user_query,
                home_status_json=env_item
            )
            
            success = result.get("success", False)
            output = result.get("output", "")
            if success:
                success_count += 1
            
        except Exception as e:
            success = False
            output = f"Exception: {str(e)}"
            result = {"success": False, "output": output, "type": "crash"}

        # 4. Record results
        log_entry = {
            "case_id": case.get('id'),
            "home_id": home_id,
            "query": user_query,
            "ground_truth_type": case_type,
            "ground_truth_output": ground_truth_output,
            "sage_output": output,
            "success": success,
            "result_type": result.get("type", "unknown")
        }
        results.append(log_entry)
        
        # Save incrementally to prevent data loss
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    print("\n" + "=" * 80)
    print(f"✅ Experiment finished!")
    print(f"📊 Total cases processed: {len(results)}")
    print(f"⏱️  Total time: {elapsed_time}")
    print(f"⚡ Average time per case: {elapsed_time.total_seconds() / len(results):.2f}s")
    
    # Print summary statistics
    print(f"📈 Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    
    # Save summary
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SAGE-HomeBench Experiment Summary (Qwen)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write(f"Model: Qwen2.5-7B-Instruct (Local)\n")
        f.write(f"Model Path: {MODEL_PATH}\n")
        f.write(f"Device: {config['device']}\n")
        f.write(f"Test cases: {len(results)}\n")
        if RANDOM_SAMPLE:
            f.write(f"Sampling: Random (seed={RANDOM_SEED})\n")
        else:
            f.write(f"Sampling: Sequential (first {TEST_LIMIT})\n")
        f.write(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)\n")
        f.write(f"Total time: {elapsed_time}\n")
        f.write(f"Average time per case: {elapsed_time.total_seconds() / len(results):.2f}s\n\n")
        f.write("=" * 80 + "\n")
        f.write("Output Files:\n")
        f.write("=" * 80 + "\n")
        f.write(f"- Results: {OUTPUT_FILE}\n")
        f.write(f"- Config: {CONFIG_FILE}\n")
        f.write(f"- Summary: {SUMMARY_FILE}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Evaluation Command:\n")
        f.write("=" * 80 + "\n")
        f.write(f"python evaluate_results.py --input {OUTPUT_FILE} --test_data {DATA_PATH}\n")
    
    print(f"\n📄 Summary saved to: {SUMMARY_FILE}")
    print(f"📄 Results saved to: {OUTPUT_FILE}")
    print("\n" + "=" * 80)
    print("📊 To evaluate results, run:")
    print(f"   python evaluate_results.py --input {OUTPUT_FILE} --test_data {DATA_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
