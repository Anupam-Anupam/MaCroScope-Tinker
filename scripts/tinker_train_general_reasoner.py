"""
This script is modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/rl_loop.py
"""
import logging
import time
import re

from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str:
    """
    Extract the last occurrence of a boxed answer from the input text.
    
    Returns:
        The content inside the last \\boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.
    
    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    
    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()
    
    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)

class GeneralVerifier:
    def __init__(self, model_name: str):
        self.llm = LLM(model=model_name, gpu_memory_utilization=0.7)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=2048)

    def _truncate_response(self, response: str) -> str:
        if response is None:
            return ""
        return self.tokenizer.decode(self.tokenizer.encode(response)[-1024:])
        
    
    def verify_batch(self, questions: list[str], ground_truths: list[str], responses: list[str]) -> list[bool]:
        student_answers = [extract_solution(response) for response in responses]
        ground_truths = [self._truncate_response(ground_truth) for ground_truth in ground_truths]
        student_answers = [self._truncate_response(student_answer) for student_answer in student_answers]
        messages = [VERIFIER_PROMPT_TEMPLATE.format(question=question, ground_truth=ground_truth, student_answer=student_answer) for question, ground_truth, student_answer in zip(questions, ground_truths, student_answers)]
        outputs = self.llm.generate(messages, sampling_params=self.sampling_params)
        verifier_responses = [output.outputs[0].text.strip() for output in outputs]
        rewards = []
        for verifier_response, ground_truth, student_answer in zip(verifier_responses, ground_truths, student_answers):
            try:
                if VERIFIER_PASS_TAG in verifier_response:
                    # penalize if student answer and ground truth having too different length
                    student_answer_length = len(self.tokenizer.encode(student_answer))
                    ground_truth_length = len(self.tokenizer.encode(ground_truth))
                    difference = abs(student_answer_length - ground_truth_length)
                    difference = min(difference, 10)
                    rewards.append(1.0 - difference * 0.05)
                else:
                    rewards.append(0.0)
            except Exception as e:
                logger.warning(f"Error verifying batch: {e}, verifier_response: {verifier_response}, ground_truth: {ground_truth}, student_answer: {student_answer}")
                rewards.append(0.0)
        return rewards

def _renderer_name_for_model(model_name: str) -> str:
    """tinker_cookbook.model_info only lists some orgs; base models use role_colon."""
    try:
        return model_info.get_recommended_renderer_name(model_name)
    except ValueError:
        return "role_colon"


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "./log"
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    dataset_path: str = "ai4collaboration/MaCroScope-Biology-Chemistry-23k"
    # HF builder config when the hub requires it (e.g. "default"); leave unset if load_dataset(path) works alone
    dataset_config: str | None = None
    train_split: str = "train"
    test_split: str = "test"
    # Sized for ~20.7k-row train split (MaCroScope-Biology-Chemistry-23k):
    # steps = floor(N / batch_size) * total_epochs (N shrinks after max_prompt_length filter).
    # Example: N=20700, batch_size=90 -> ~230 batches/epoch * 5 epochs = ~1150 steps.
    batch_size: int = 90
    val_batch_size: int = 200         # test split is ~2.3k rows; keep eval lighter than GR's 500
    group_size: int = 8               # actor_rollout_ref.rollout.n
    learning_rate: float = 5e-7       # actor_rollout_ref.actor.optim.lr (GR-recommended)
    max_prompt_length: int = 1024     # data.max_prompt_length
    max_tokens: int = 8192            # data.max_response_length
    temperature: float = 1.0          # actor_rollout_ref.rollout.temperature
    # PPO clip — GR uses clip_ratio=0.3, mapped to (1-0.3, 1+0.3) thresholds
    clip_low_threshold: float = 0.7
    clip_high_threshold: float = 1.3
    total_epochs: int = 5             # <10; with batch_size=90 -> ~1150 steps at N≈20.7k
    save_freq: int = 50               # ~10 checkpoints per ~500-step run
    test_freq: int = 25               # eval ~20x per run (was every 5 of 240+ steps)
    lora_rank: int = 32
    # W&B logging — only used if WANDB_API_KEY is exported in the environment;
    # otherwise ml_log.setup_logging silently skips wandb.
    wandb_project: str | None = "MaCroScope-Tinker"
    wandb_name: str | None = "nemotron30b-biochem23k-b90-e5"


def main(config: Config):
    import os as _os
    wandb_key = _os.environ.get("WANDB_API_KEY", "")
    use_wandb = bool(wandb_key) and wandb_key != "REPLACE_ME"
    if not use_wandb:
        # Make sure ml_log doesn't try to init wandb on a placeholder/empty key.
        _os.environ.pop("WANDB_API_KEY", None)

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project if use_wandb else None,
        wandb_name=config.wandb_name if use_wandb else None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = _renderer_name_for_model(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    verifier = GeneralVerifier("TIGER-Lab/general-verifier")

    logger.info("Loading dataset...")
    ds_kwargs: dict = {}
    if config.dataset_config is not None:
        ds_kwargs["name"] = config.dataset_config
    dataset = datasets.load_dataset(config.dataset_path, **ds_kwargs)
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset[config.train_split]
    test_dataset = dataset[config.test_split] if config.test_split in dataset else None

    def _filter_overlong(row):
        message = [
            {"role": "user", "content": row["question"] + " Please reason step by step, and put your final answer within \\boxed{}."}
        ]
        n_tokens = len(renderer.build_generation_prompt(message).to_ints())
        return n_tokens <= config.max_prompt_length

    pre = len(train_dataset)
    train_dataset = train_dataset.filter(_filter_overlong)
    logger.info(f"Filtered overlong train prompts: {pre} -> {len(train_dataset)} (max_prompt_length={config.max_prompt_length})")
    if test_dataset is not None:
        pre_t = len(test_dataset)
        test_dataset = test_dataset.filter(_filter_overlong)
        logger.info(f"Filtered overlong test prompts: {pre_t} -> {len(test_dataset)}")

    n_train_batches_per_epoch = max(1, len(train_dataset) // config.batch_size)
    n_total_batches = n_train_batches_per_epoch * config.total_epochs

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(
        f"Training for {n_total_batches} batches "
        f"({config.total_epochs} epochs x {n_train_batches_per_epoch} batches); "
        f"train_rows={len(train_dataset)}, batch_size={config.batch_size}"
    )

    def _run_eval(global_step: int, eval_sampling_client) -> None:
        if test_dataset is None or len(test_dataset) == 0:
            return
        n_eval = min(config.val_batch_size, len(test_dataset))
        eval_rows = test_dataset.select(range(n_eval))
        eval_questions: list[str] = []
        eval_answers: list[str] = []
        eval_responses: list[str] = []
        for question, answer in zip(eval_rows["question"], eval_rows["answer"]):
            message = [
                {"role": "user", "content": question + " Please reason step by step, and put your final answer within \\boxed{}."}
            ]
            model_input = renderer.build_generation_prompt(message)
            sample_result = eval_sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            ).result()
            sampled_tokens = sample_result.sequences[0].tokens
            parsed_message, _ = renderer.parse_response(sampled_tokens)
            eval_questions.append(question)
            eval_answers.append(answer)
            eval_responses.append(parsed_message["content"])
        eval_rewards = verifier.verify_batch(eval_questions, eval_answers, eval_responses)
        ml_logger.log_metrics(
            {"eval/reward_mean": sum(eval_rewards) / max(1, len(eval_rewards)),
             "eval/n": float(len(eval_rewards))},
            step=global_step,
        )

    #  Main training loop
    for batch_idx in range(start_batch, n_total_batches):
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        epoch = batch_idx // n_train_batches_per_epoch
        within_epoch_idx = batch_idx % n_train_batches_per_epoch
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "progress/epoch": epoch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_total_batches,
        }

        # Save checkpoint
        if step % config.save_freq == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = within_epoch_idx * config.batch_size
        batch_end = min((within_epoch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Periodic eval on the test split (mirrors trainer.test_freq)
        if config.test_freq > 0 and step % config.test_freq == 0:
            _run_eval(step, sampling_client)
        # Set up sampling parameters

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts: list[list[int]] = []
        
        # Step 1: Generate all samples
        for question in batch_rows["question"]:
            message = [
                {"role": "user", "content": question + " Please reason step by step, and put your final answer within \\boxed{}."}
            ]
            model_input = renderer.build_generation_prompt(message)
            prompt_tokens = model_input.to_ints()

            # Generate response
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_prompts.append(prompt_tokens)

        # Step 2: Collect all responses and prepare for verification
        all_questions: list[str] = []
        all_answers: list[str] = []
        all_responses: list[str] = []
        all_metadata: list[dict] = []  # Store metadata for reconstruction
        
        for sample_futures, prompt_tokens, question, answer in zip(
                batch_futures, batch_prompts, batch_rows["question"], batch_rows["answer"]
        ):
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            group_responses: list[str] = []
            
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                response_content = parsed_message["content"]
                group_responses.append(response_content)
                
                # Add to batch-level lists for verification
                all_questions.append(question)
                all_answers.append(answer)
                all_responses.append(response_content)
            
            # Store metadata for this group
            all_metadata.append({
                "group_tokens": group_tokens,
                "group_logprobs": group_logprobs,
                "group_ob_lens": group_ob_lens,
                "group_size": len(group_responses),
                "question": question,
                "answer": answer
            })
        
        # Step 3: Call verifier once for entire batch
        all_rewards = verifier.verify_batch(all_questions, all_answers, all_responses)
        
        # Step 4: Process rewards and create training datums
        reward_idx = 0
        for metadata in all_metadata:
            group_size = metadata["group_size"]
            group_rewards = all_rewards[reward_idx:reward_idx + group_size]
            reward_idx += group_size
            
            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # Check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip question because all advantages are the same
                continue

            for tokens, logprob, advantage, ob_len in zip(
                metadata["group_tokens"], 
                metadata["group_logprobs"], 
                advantages, 
                metadata["group_ob_lens"]
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        # Training step (PPO with clip thresholds mirroring GR clip_ratio=0.3)
        fwd_bwd_future = training_client.forward_backward(
            training_datums,
            loss_fn="ppo",
            loss_fn_config={
                "clip_low_threshold": config.clip_low_threshold,
                "clip_high_threshold": config.clip_high_threshold,
            },
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics[]
        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards)
        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_total_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)