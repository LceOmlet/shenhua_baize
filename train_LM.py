from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only, get_chat_template
from transformers import AutoTokenizer, DefaultDataCollator
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset
import torch
import argparse
import re
import os
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from unsloth import apply_chat_template


# 使用特定的标记来标示思考和回答部分
THINK_TAG = '<|think|>'
ANSWER_TAG = '<\|think|>'
END_TAG = '<|im_end|>'


def message_construct(sample, processor):
    # Define system message
    system_message = {"role": "system", "content": "You are a helpful assistant that provides clear, accurate, and thoughtful responses."}
    
    # Define user message
    user_message = {"role": "user", "content": sample['question']}
    
    # Define conversation with system and user roles
    conversation = [system_message, user_message]

    # 为助手回答构建完整的思考和回答文本 - 去掉END_TAG
    assistant_text = f"{THINK_TAG}\n{sample['gemini_thinking_trajectory']}\n{ANSWER_TAG}\n{sample['gemini_attempt']}"

    assistant_message = {"role": "assistant", "content": assistant_text}

    conversation.append(assistant_message)

    return conversation

def set_custom_template(tokenizer, template_path=None):
    """设置自定义的聊天模板"""
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template_content = f.read()
            
        tokenizer.chat_template = template_content
        print(f"已加载自定义模板: {template_path}")
    else:
        print("使用默认的Gemma-3模板")
        
    return tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using Unsloth's FastModel")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-3B-Instruct", 
                        help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=4096 + 512, 
                        help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, 
                        help="Load model in 4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, 
                        help="Load model in 8-bit quantization")
    parser.add_argument("--full_finetuning", action="store_true", default=False, 
                        help="Perform full finetuning instead of LoRA")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=8, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.0, 
                        help="LoRA dropout probability")
    
    # Training parameters
    parser.add_argument("--dataset_path", type=str, default="simplescaling/s1K_tokenized", 
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./s1", 
                        help="Output directory for saving model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, 
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="Initial learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="Maximum number of training steps. Overrides num_train_epochs if set")
    parser.add_argument("--warmup_steps", type=int, default=5, 
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, 
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=100, 
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=3, 
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--seed", type=int, default=3407, 
                        help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, 
                        help="Use bfloat16 precision")
    
    # Save options
    parser.add_argument("--save_model", action="store_true", default=True, 
                        help="Save the final model")
    parser.add_argument("--save_as_full_model", action="store_true", default=False, 
                        help="Save as a full model instead of LoRA adapters only")
    parser.add_argument("--push_to_hub", action="store_true", default=False, 
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                        help="Hub model ID if pushing to hub")
    parser.add_argument("--hub_token", type=str, default=None, 
                        help="Hugging Face token for pushing to hub")
    
    return parser.parse_args()

def test_sample(processor, dataset):
    """测试样本数据的格式化和tokenization"""
    if len(dataset) == 0:
        print("Dataset is empty")
        return
    
    sample = dataset[0]
    print(f"\n测试样本数据:")
    print(f"问题: {sample['question'][:50]}...")
    print(f"思考: {sample['gemini_thinking_trajectory'][:50000]}...")
    print(f"回答: {sample['gemini_attempt'][:50000]}...")
    
    try:
        # 测试用户消息格式化
        user_message = {"role": "user", "content": sample['question']}
        prompt = processor.apply_chat_template([user_message], tokenize=False, add_generation_prompt=True)
        print(f"\n格式化后的用户提示:")
        print(f"{prompt[:100]}...")
        
        # 测试完整消息格式化 - 使用正确的标记格式
        assistant_text = f"{THINK_TAG}\n{sample['gemini_thinking_trajectory']}\n{ANSWER_TAG}\n{sample['gemini_attempt']}"
        assistant_message = {"role": "assistant", "content": assistant_text}
        full_text = processor.apply_chat_template([user_message, assistant_message], tokenize=False)
        print(f"\n格式化后的完整文本:")
        print(f"{full_text[:100]}...")
        
        # 测试tokenization
        encoded = processor(prompt, return_tensors="pt")
        print(f"\nTokenization结果:")
        print(f"Token IDs shape: {encoded.input_ids.shape}")
        print(f"First few tokens: {encoded.input_ids[0][:10].tolist()}")
        
        return True
    except Exception as e:
        print(f"测试样本时出错: {e}")
        print("尝试使用processor的其他方法...")
        try:
            if hasattr(processor, 'tokenizer'):
                print("\n使用内部tokenizer:")
                encoded = processor.tokenizer(sample['question'], return_tensors="pt")
                print(f"Token IDs shape: {encoded.input_ids.shape}")
            elif hasattr(processor, 'text_processor') and hasattr(processor.text_processor, 'tokenizer'):
                print("\n使用text_processor.tokenizer:")
                encoded = processor.text_processor.tokenizer(sample['question'], return_tensors="pt")
                print(f"Token IDs shape: {encoded.input_ids.shape}")
            else:
                print("无法找到合适的tokenizer进行测试")
        except Exception as nested_e:
            print(f"再次尝试时出错: {nested_e}")
        
        return False

# 添加函数来注册特殊标记到tokenizer
def add_special_thinking_tokens(processor):
    """将思考和回答标记添加为特殊token"""
    # 对于Gemma3Processor，需要访问内部的tokenizer
    try:
        # 尝试直接访问tokenizer属性
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            special_tokens_dict = {
                'additional_special_tokens': [THINK_TAG, ANSWER_TAG]
            }
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"添加了 {num_added_toks} 个特殊token到tokenizer中")
            
            # 打印特殊token的ID
            for token in [THINK_TAG, ANSWER_TAG]:
                token_id = tokenizer.convert_tokens_to_ids(token)
                print(f"Token '{token}' ID: {token_id}")
                
            # 更新processor中的tokenizer
            processor.tokenizer = tokenizer
        else:
            # 如果没有tokenizer属性，可能是其他属性
            print("注意: processor没有tokenizer属性，尝试其他方法")
            # 检查是否为Gemma3Processor
            tokenizer = processor
            special_tokens_dict = {
                'additional_special_tokens': [THINK_TAG, ANSWER_TAG]
            }
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"通过text_processor添加了 {num_added_toks} 个特殊token")
            
            # 打印特殊token的ID
            for token in [THINK_TAG, ANSWER_TAG]:
                token_id = tokenizer.convert_tokens_to_ids(token)
                print(f"Token '{token}' ID: {token_id}")
                    
                # 更新processor中的tokenizer
            else:
                print("警告: 无法访问或修改tokenizer，特殊标记可能无法正确处理")
                print(f"Processor类型: {type(processor)}")
                print(f"可用属性: {dir(processor)}")
    except Exception as e:
        print(f"添加特殊标记时出错: {e}")
        print(f"Processor类型: {type(processor)}")
        print(f"可用属性: {dir(processor)}")
    
    return processor

def test_special_tokens(tokenizer):
    """测试特殊标记是否被正确识别"""
    print("\n测试特殊标记:")
    
    # 测试单个标记的tokenization
    for token in [THINK_TAG, ANSWER_TAG]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        print(f"Token '{token}' 被编码为: {ids}")
        print(f"解码回来: '{tokenizer.decode(ids)}'")
    
    # 测试在上下文中的标记
    test_text = f"这是一个测试 {THINK_TAG} 思考内容 {ANSWER_TAG} 回答内容"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n测试文本: '{test_text}'")
    print(f"编码后的token IDs: {encoded}")
    print(f"解码后: '{decoded}'")
    
    return True

if __name__ == "__main__":
    args = parse_args()
    
    # Load model using Unsloth's FastModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        load_in_8bit = args.load_in_8bit,
        full_finetuning = args.full_finetuning,
    )
    
    # 打印tokenizer的类型，帮助调试
    print(f"\nTokenizer类型: {type(tokenizer)}")
    print(f"Tokenizer属性: {dir(tokenizer)[:10]} ... 等")
    
    # 注册特殊思考和回答标记到tokenizer
    tokenizer = add_special_thinking_tokens(tokenizer)
    
    # 调整模型embedding大小以适应新添加的特殊token
    try:
        if hasattr(tokenizer, 'tokenizer'):
            vocab_size = len(tokenizer.tokenizer)
        elif hasattr(tokenizer, 'text_processor') and hasattr(tokenizer.text_processor, 'tokenizer'):
            vocab_size = len(tokenizer.text_processor.tokenizer)
        else:
            vocab_size = len(tokenizer)
        
        model.resize_token_embeddings(vocab_size)
        print(f"已调整模型embedding大小至: {vocab_size}")
    except Exception as e:
        print(f"调整模型embedding大小时出错: {e}")

    
    # Parse raw dataset to extract thinking and answer sections
    def parse_raw_dataset(dataset, split="train"):
        parsed_data = []
        for item in dataset[split]:
            text = item["text"]
            
            # Extract user question
            user_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
            user_match = re.search(user_pattern, text, re.DOTALL)
            
            # Extract assistant thinking and answer
            assistant_pattern = r"<\|im_start\|>assistant\n<\|im_start\|>think\n(.*?)<\|im_start\|>answer\n(.*?)<\|im_end\|>"
            assistant_match = re.search(assistant_pattern, text, re.DOTALL)
            
            if user_match and assistant_match:
                question = user_match.group(1).strip()
                thinking = assistant_match.group(1).strip()
                answer = assistant_match.group(2).strip()
                
                parsed_data.append({
                    "question": question,
                    "gemini_thinking_trajectory": thinking, 
                    "gemini_attempt": answer
                })
        
        return Dataset.from_list(parsed_data)
    
    # Process dataset using raw data
    raw_dataset = load_dataset(args.dataset_path)
    processed_dataset = parse_raw_dataset(raw_dataset)
    
    # Test sample processing
    print(f"Dataset size: {len(processed_dataset)}")
    test_sample(tokenizer, processed_dataset)
    
    # Create training dataset
    def apply_chat_template(sample):
            
        # Construct prompt and assistant text using message_construct
        conversation = message_construct(sample, tokenizer)

        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        return {'text':text}
    
    train_dataset = processed_dataset.map(apply_chat_template, batched=False)
    # Create the training dataset
    
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = args.seed,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Setup SFT trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        args = SFTConfig(
            output_dir = args.output_dir,
            per_device_train_batch_size = args.per_device_train_batch_size,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            learning_rate = args.learning_rate,
            num_train_epochs = args.num_train_epochs,
            max_steps = args.max_steps,
            warmup_steps = args.warmup_steps,
            weight_decay = args.weight_decay,
            logging_steps = args.logging_steps,
            save_strategy = "steps",
            save_steps = args.save_steps,
            save_total_limit = args.save_total_limit,
            bf16 = args.bf16,
            seed = args.seed,
            report_to = "none",
            optim = "adamw_8bit",
            lr_scheduler_type = "linear",
            # data_collator = DefaultDataCollator(),
        ),
        packing = False,  # Don't pack sequences - important for thinking/answer format
    )

    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>system", response_part="<|im_start|>assistant")
    
    # 在SFTDataset中已设置了标签掩码(-100)，同时使用了system/user/assistant角色格式
    # train_on_responses_only确保仅对助手部分计算损失
    
    # Print GPU memory usage before training
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0
    print(f"Memory reserved before training: {start_gpu_memory} GB")
    
    # Start training
    print("Starting training...")
    trainer_stats = trainer.train()
    
    # Print training stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3) if torch.cuda.is_available() else 0
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    
    # Save the model
    if args.save_model:
        if args.save_as_full_model:
            model.save_pretrained_merged(args.output_dir, tokenizer)
        else:
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        
        print(f"Model saved to {args.output_dir}")
        
        if args.push_to_hub and args.hub_model_id and args.hub_token:
            if args.save_as_full_model:
                model.push_to_hub_merged(
                    args.hub_model_id,
                    tokenizer,
                    token = args.hub_token
                )
            else:
                model.push_to_hub(args.hub_model_id, token=args.hub_token)
                tokenizer.push_to_hub(args.hub_model_id, token=args.hub_token)
            
            print(f"Model pushed to Hugging Face Hub: {args.hub_model_id}")
