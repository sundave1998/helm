from copy import deepcopy
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence
from .huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import HuggingFaceModelConfig, get_huggingface_model_config

import traceback
import os
import threading


def print_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    print(''.join(lines))

from multiprocessing import Value
from ctypes import c_int


class HuggingFaceServer:

    CUDA_CNT = Value(c_int, 0)

    def __init__(self, model_config: HuggingFaceModelConfig):
        if torch.cuda.is_available():
            with HuggingFaceServer.CUDA_CNT.get_lock():
                # TODO: support cuda in the future
                # self.device: str = f"cuda:{HuggingFaceServer.CUDA_CNT.value}"
                self.device = "cuda:0"
                # HuggingFaceServer.CUDA_CNT.value += 1
            hlog(f"CUDA is available, initializing with {self.device}...")
        else:
            self.device = "cpu"
        model_kwargs = {}
        if model_config.revision:
            model_kwargs["revision"] = model_config.revision
        print("current model config is as follows:", model_config)
        with htrack_block(f"Loading Hugging Face model for config {model_config}"):
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_config.model_id, trust_remote_code=True, **model_kwargs
            # ).to(self.device)
            import sys
            # import os
            # file_dir = os.path.join(os.path.dirname(__file__), '..')
            # file_dir = os.path.join(file_dir, '..')
            # file_dir = os.path.join(file_dir, '..')
            # file_dir = os.path.join(file_dir, '..')
            file_dir = '/mnt/data/youbangsun/FederatedScope'
            sys.path.append(file_dir)
            from federatedscope.core.configs.config import global_cfg, CfgNode
            # from federatedscope.core.auxiliaries.model_builder import get_model
            init_cfg = global_cfg.clone()

            init_cfg.merge_from_file("/mnt/data/youbangsun/FederatedScope/llama_DP.yaml")
            init_cfg.wandb.use = False
            init_cfg.nbafl.use_opacus = False
            init_cfg.nbafl.use = False
            
            # self.model = get_model(client_specific_config)
            print("configs loaded, getting llm using fedscope")
            try:

                from federatedscope.llm.model import get_llm
                self.model = get_llm(init_cfg).to(self.device)
            except:
                print("getllm failed")
            try:
                def load_model(model, path):
                    if os.path.exists(path):
                        ckpt = torch.load(path, map_location=self.device)
                        model.load_state_dict(ckpt['model'])
                        print("checkpoint loaded, current iteration is:", ckpt['cur_round'])
                        return model
                ckpt_path = "/mnt/data/youbangsun/FederatedScope/llama.ckpt"
                self.model = load_model(self.model, ckpt_path)
            except:
                print("load checkpoint error!!")
            
            
            
        hlog(f"{type(self.model)}, {self.model.__class__}")
        with htrack_block(f"Loading Hugging Face tokenizer model for config {model_config}"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_id, **model_kwargs)

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt").to(self.device)
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(raw_request["stop_sequences"], return_token_type_ids=False)
            assert len(stop_sequence_ids.input_ids) == 1, f"Total number of stop words should be 1 rather than {stop_sequence_ids.input_ids}"
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            # if key not in ["engine", "prompt", "echo_prompt"]
            # TODO: for the stop sequences bug
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # dawei: resolve the bug for huggingface
        # Use HuggingFace's `generate` method.
        print("encoded_input", encoded_input)
        # if encoded_input.has_key('token_type_ids'):
        #     del encoded_input['token_type_ids']
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        # TODO: Get rid of the extra tokenization?
        all_tokens = [self.tokenizer.convert_ids_to_tokens(sequence) for sequence in sequences]
        all_tokens = [
            [self.tokenizer.convert_tokens_to_string([token]) for token in sequence_tokens]
            for sequence_tokens in all_tokens
        ]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for (decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts) in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}

    def get_model_server_instance(self, model) -> HuggingFaceServer:
        if model not in self.model_server_instances:
            model_config = get_huggingface_model_config(model)
            if model_config:
                self.model_server_instances[model] = HuggingFaceServer(model_config)
            elif model == "EleutherAI/gpt-j-6B":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("EleutherAI/gpt-j-6B")
                )
            elif model == "huggingface/gpt2":
                self.model_server_instances[model] = HuggingFaceServer(HuggingFaceModelConfig.from_string("gpt2"))
            elif model == "bigcode/santacoder":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("bigcode/santacoder")
                )
            elif model == "huggingface/starcoder":
                self.model_server_instances[model] = HuggingFaceServer(
                    HuggingFaceModelConfig.from_string("bigcode/starcoder")
                )
            else:
                raise Exception(f"Unknown HuggingFace model: {model}")

        return self.model_server_instances[model]

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        try:

            def do_it():
                return model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {repr(e)}"
            print_traceback(e)
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                    else:
                        tokens = tokenizer.tokenize(request.text)
                        # TODO(1522): Reenable this to revove "▁"
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {repr(e)}"
            print_traceback(e)
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {repr(e)}"
            print_traceback(e)
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
