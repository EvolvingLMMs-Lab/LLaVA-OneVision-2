""" Mcore_checkpoint converter for aiak megatron. """
import io
import os
from typing import Tuple, Literal
import shutil
import torch
import types
from tqdm import tqdm

import resource

import concurrent.futures
from convert_checkpoint.arguments import parse_args
from convert_checkpoint.abstact_checkpoint import AbstractCheckpoint
from convert_checkpoint.common_checkpoint import CommonCheckpoint
from convert_checkpoint.megatron_optimizer import MegatronOptimizer, merge_optimizer_by_pp_tp, merge_optimizer_by_dp
from convert_checkpoint.utils import (
    check_path_in_dict,
    get_element_from_dict_by_path,
    add_embedding_padding, cut_embedding_padding,
    transpose_shape0,
    partition_balanced,
    custom_partition_imbalanced,
    uneven_vpp_partition,
    touch_file,
    check_all_done,
    get_done_keys,
    ceil_div,
)

from convert_checkpoint import utils

from convert_checkpoint.common_checkpoint import (
    WORD_EMBEDDINGS,
    WORD_POSITION_EMBEDDINGS,
    WORD_BLOCK_POSITION_EMBEDDINGS,
    TRANSFORMER,
    LAYER_PREFIX,
    MTP_LAYER_PREFIX,
    INPUT_LAYERNORM,
    ATTENTION_ROTARY_EMB_INV_FREQ,
    ATTENTION_QUERY_KEY_VALUE,
    ATTENTION_QKV_MAP,
    ATTENTION_DENSE,
    POST_ATTENTION_LAYERNORM,
    MOE_GATE,
    MOE_GATE_BIAS,
    MOE_MLP,
    MOE_EXPERT,
    MOE_GROUPED_GEMM_EXPERT,
    MOE_SHARED_EXPERT,
    MLP_DENSE_H_TO_4H,
    MLP_DENSE_4H_TO_H,
    POST_MLP_LAYERNORM,
    FINAL_LAYERNORM,
    WORD_EMBEDDINGS_FOR_HEAD,
    WORD_EMBEDDINGS_TPL,
    WORD_POSITION_EMBEDDINGS_TPL,
    WORD_BLOCK_POSITION_EMBEDDINGS_TPL,
    TRANSFORMER_TPL,
    WORD_EMBEDDINGS_FOR_HEAD_TPL,
    MTP_WORD_EMBEDDING,
    MTP_ENORM,
    MTP_HNORM,
    MTP_EH_PROJ,
    MTP_SHARED_HEAD_NORM,
    MTP_SHARED_HEAD_HEAD,
)


def get_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Mcore-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


class McoreCheckpoint(AbstractCheckpoint):
    """
        McoreCheckpoint
    """

    def __init__(self, num_layers, load_path=None):
        super().__init__(num_layers)
        self.version = 3.0
        self.iteration = 0
        self.args = parse_args()
        self.rng_state = None
        self.load_path = load_path

    def init_pipeline_size(self, pp, tp, dp, ep, tensor_parallel_dim, stage, num_layers_for_test=None,
                           custom_pipeline_layers=None, etp=None):
        "Initialize tp pp dp"
        self.pp = pp
        self.tp = tp
        self.dp = dp
        self.ep = ep
        self.etp = etp
        self.tensor_parallel_dim = tensor_parallel_dim
        self.num_stages = stage or 1
        args = parse_args()
        if args.decoder_first_pipeline_num_layers is not None and args.decoder_last_pipeline_num_layers is not None:
            assert (self.num_layers - args.decoder_first_pipeline_num_layers - \
                    args.decoder_last_pipeline_num_layers) // (self.pp - 2) % self.num_stages == 0
        elif num_layers_for_test is None and custom_pipeline_layers is None:
            assert self.num_layers // self.pp % self.num_stages == 0

    def get_state_dict(self, p, t, e=None):
        if self.load_path is None:
            if utils.LOADED_STATE_DICT is None:
                return None
            if t == 0 and e is None:
                if p in utils.LOADED_STATE_DICT:
                    return utils.LOADED_STATE_DICT[p]
                else:
                    return None
            elif t == 0 and e is not None:
                if p in utils.LOADED_STATE_DICT and e in utils.LOADED_STATE_DICT[p]:
                    return utils.LOADED_STATE_DICT[p][e]
                else:
                    return None
            raise Exception("get_state_dict failed. No load_path specified.")
        checkpoint_name = "model_optim_rng.pt"
        if e is None or self.ep == 1:
            sub_dir_name = f"mp_rank_{t:02d}" if self.pp == 1 \
                    else f"mp_rank_{t:02d}_{p:03d}"
            checkpoint_path = os.path.join(self.load_path, sub_dir_name, checkpoint_name)
            if not os.path.exists(checkpoint_path) and self.pp == 1:
                # Fallback: checkpoint may be stored in EP-sharded layout (mp_rank_{t}_{e})
                # even when this component doesn't use EP (e.g. ViT in an MoE checkpoint).
                # Use ep=0 shard — non-EP weights are replicated across all EP ranks.
                fallback_dir = f"mp_rank_{t:02d}_000"
                fallback_path = os.path.join(self.load_path, fallback_dir, checkpoint_name)
                if os.path.exists(fallback_path):
                    print(f"load checkpoint (ep=0 fallback): {fallback_path}")
                    return torch.load(fallback_path, map_location="cpu", weights_only=False)
            print(f"load checkpoint: {checkpoint_path}")
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            sub_dir_name = f"mp_rank_{t:02d}_{e:03d}" if self.pp == 1 \
                else f"mp_rank_{t:02d}_{p:03d}_{e:03d}"
            checkpoint_path = os.path.join(self.load_path, sub_dir_name, checkpoint_name)
            print(f"load checkpoint: {checkpoint_path}")
            return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def init_optimizer(self, use_distributed_optimizer):
        assert self.pp > 0 and self.tp > 0
        self.use_distributed_optimizer = use_distributed_optimizer
        self.optim_state_dict = []
        for p in range(self.pp):
            self.optim_state_dict.append([])
            if self.ep is None:
                for t in range(self.tp):
                    opt = MegatronOptimizer.generate_optimizer(self, self.num_layers // self.pp, p)
                    self.optim_state_dict[p].append(opt)
            else:
                for e in range(self.ep):
                    self.optim_state_dict[p].append([])
                    for t in range(self.tp):
                        opt = MegatronOptimizer.generate_optimizer(self, self.num_layers // self.pp, p)
                        self.optim_state_dict[p][e].append(opt)

    def init_layers(self):
        self.layers = []
        for p in range(self.pp):
            self.layers.append(self.get_layers(p))

    def init_named_parameters_shape(self):
        self.named_parameters_shape = self._get_named_parameters_shape()
        self.named_parameters_shape_by_p = [None] * self.pp
        self.named_parameters_shape_by_t = [None] * self.tp
        self.named_parameters_shape_by_pt = []
        for p in range(self.pp):
            self.named_parameters_shape_by_p[p] = self._get_named_parameters_shape(pp_rank=p)
        for t in range(self.tp):
            self.named_parameters_shape_by_t[t] = self._get_named_parameters_shape(tp_rank=t)
        for p in range(self.pp):
            self.named_parameters_shape_by_pt.append([None] * self.tp)
            for t in range(self.tp):
                self.named_parameters_shape_by_pt[p][t] =  self._get_named_parameters_shape(p, t)

    @staticmethod
    def get_virtual_partition(dualpipev, stage_index, p, pp, num_layers_in_rank, m_ckpt):
        if dualpipev:
            if stage_index == 0:
                virtual_p = p
            else:
                virtual_p = pp * stage_index + (pp - 1 - p)
        else:
            virtual_p = p + pp * stage_index
        layer_offset = sum(num_layers_in_rank[:virtual_p])
        transformer = m_ckpt.get_transformer_name(stage_index)
        return virtual_p, layer_offset, transformer

    @staticmethod
    def convert_from_common(c_ckpt, c_config, save_path=None, m_config=None, save_optim=True):
        """
        Convert common checkpoint to mcore checkpoint.

        Args:
            c_ckpt: CommonCheckpoint
            c_config: CommonConfig
        """

        print("\n==================== Common -> Mcore ====================")

        name_map = c_config.get("name_map")["mcore"]
        cargs = c_config.get_args("common")
        hargs = c_config.get_args("huggingface")
        margs = c_config.get_args("mcore")
        dtype = c_config.get_dtype()
        tensor_parallel_dim = c_config.get("tensor_parallel_dim")
        separate_dtype = c_config.get("separate_dtype")

        args = parse_args()
        cache_path = args.cache_path
        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        dp = args.data_parallel_size
        ep = args.expert_parallel_size
        etp = args.expert_tensor_parallel_size if hasattr(args, 'expert_tensor_parallel_size') else None
        dualpipev = args.vpp_scheduler == 'dualpipev'
        custom_pipeline_layers = args.custom_pipeline_layers
        num_experts = args.num_experts
        if num_experts is not None:
            assert num_experts > 0, "num_experts must be greater than zero"
            if ep is None:
                ep = 1  # if ep is not set, will process the model as no MoE
        num_experts_for_test = args.num_experts_for_test
        num_nextn_predict_layers = hargs.get("num_nextn_predict_layers", 0)
        ori_num_layers = cargs["num_layers"]
        num_layers = ori_num_layers + num_nextn_predict_layers
        hidden_size = cargs["hidden_size"]
        use_distributed_optimizer = margs["use_distributed_optimizer"]
        num_layers_per_stage = margs["num_layers_per_virtual_pipeline_stage"]
        if num_layers_per_stage:
            stage = num_layers // pp // num_layers_per_stage
        else:
            stage = args.num_virtual_stages_per_pipeline_rank or 1
        num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            assert args.num_virtual_stages_per_pipeline_rank is not None, "num_virtual_stages_per_pipeline_rank is required"
        ignore_tp_keys = margs.get("ignore_tp_keys", [])
        pretrain_as_fp8 = args.pretrain_as_fp8

        # get specific layer for test
        layer_for_test = args.layer_for_test
        num_layers_for_test = None
        if layer_for_test is not None:
            splits = []
            if layer_for_test.find(',') != -1:
                splits = [int(s) for s in layer_for_test.split(',')]
            else:
                splits = [int(layer_for_test)]
            num_layers_for_test = len(splits)
            layer_for_test_map = {}
            for index, layer_id in enumerate(splits):
                layer_for_test_map[layer_id] = index

        m_ckpt = McoreCheckpoint(num_layers)
        m_ckpt.set_dtype(dtype)
        m_ckpt.init_pipeline_size(pp, tp, dp, ep, tensor_parallel_dim, stage, num_layers_for_test=num_layers_for_test,
                                  custom_pipeline_layers=custom_pipeline_layers, etp=etp)
        m_ckpt.set_name_map(name_map)

        m_ckpt.iteration = c_ckpt.other_args.get("iteration", m_ckpt.iteration)
        m_ckpt.version = c_ckpt.other_args.get("version", m_ckpt.version)
        m_ckpt.args = c_ckpt.other_args.get("args", m_ckpt.args)
        m_ckpt.rng_state = c_ckpt.other_args.get("rng_state", m_ckpt.rng_state)

        if c_ckpt.has_optimizer() and ep is None:
            # 1. optimizer
            m_ckpt.init_layers()
            m_ckpt.init_named_parameters_shape()
            m_ckpt.init_optimizer(use_distributed_optimizer)
            opt = MegatronOptimizer.generate_optimizer(m_ckpt, c_ckpt.num_layers)
            opt.load(c_ckpt.state_dict["optimizer"])
            if margs.get("add_embedding_padding", False):
                divisible_by = margs["make_vocab_size_divisible_by"]
                vocab_size = cargs["vocab_size"]
                padded_vocab_size = margs.get("pad_vocab_size_to")
                opt.add_embedding_padding(divisible_by, vocab_size, tp, hidden_size, padded_vocab_size)
            if m_ckpt.pp == 1 and not margs.get("untie_embeddings_and_output_weights", False):
                opt.remove_word_embedding_for_head()
            opt.build_param_map(m_ckpt.get_named_parameters_shape())
            if stage > 1:
                opt.interleave(stage, pp)
            opts = opt.chunk_by_pp_tp(pp, tp, margs)
            for p in range(pp):
                for t in range(tp):
                    named_parameters_shape = m_ckpt.get_named_parameters_shape(p, t)
                    opts[p][t].build_param_map(named_parameters_shape)
            m_ckpt.optim_state_dict = opts
        else:
            m_ckpt.optim_state_dict = None
            print("> optimizer empty")

        layer_prefix = name_map[LAYER_PREFIX]
        if num_nextn_predict_layers > 0:
            min_mtp_layer_id = ori_num_layers
            mtp_layer_prefix = name_map[MTP_LAYER_PREFIX]
        else:
            min_mtp_layer_id = None
        if custom_pipeline_layers is not None:
            assert num_layers_in_first_pipeline_stage is None and num_layers_in_last_pipeline_stage is None, \
                "custom_pipeline_layers need not num_layers_in_first_pipeline_stage or in_last_pipeline_stage"
            num_layers_in_rank, _ = custom_partition_imbalanced(ori_num_layers, pp * stage, custom_pipeline_layers)
            num_layers_in_rank[-1] += num_nextn_predict_layers
        elif num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            num_layers_in_rank = uneven_vpp_partition(num_layers, pp, stage, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage)
            num_layers_in_rank[-1] += num_nextn_predict_layers
        else:
            num_layers_in_rank, _ = partition_balanced(num_layers, pp * stage)
        first_k_dense_replace = margs.get("first_k_dense_replace", None)

        done_dir = os.path.join(save_path, "dones")
        need_check_dones = False
        if args.resume_convert and os.path.exists(done_dir):
            need_check_dones = True
            done_keys = get_done_keys(done_dir, pp, ep)
        else:
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
        release_dir, save_margs = m_ckpt.pre_save(save_path, m_config)
        if layer_for_test is not None:
            save_margs.num_layers = len(layer_for_test_map) - num_nextn_predict_layers
        os.makedirs(done_dir, exist_ok=True)

        if ep is not None:
            post_attention_layernorm = name_map[POST_ATTENTION_LAYERNORM]
            if MOE_MLP in name_map:
                moe_mlp = name_map[MOE_MLP]
            moe_expert = name_map[MOE_EXPERT]
            if MOE_GROUPED_GEMM_EXPERT in name_map:
                moe_grouped_gemm_expert = name_map[MOE_GROUPED_GEMM_EXPERT]
            if MOE_SHARED_EXPERT in name_map:
                moe_shared_expert = name_map[MOE_SHARED_EXPERT]
            if num_experts_for_test is None:
                experts_ids = [x for x in range(num_experts)]
                chunks = [experts_ids[x:x + num_experts // ep]
                    for x in range(0, len(experts_ids), num_experts // ep)] # ep_id -> [expert_ids]
            else:
                experts_ids = [x for x in range(num_experts_for_test)]
                chunks = [experts_ids[x:x + num_experts_for_test // ep]
                    for x in range(0, len(experts_ids), num_experts_for_test // ep)] # ep_id -> [expert_ids]

            expert_local_mapping = {}
            expert_ep_mapping = {}
            ep_expert_mapping = {}
            for ep_id, chunk in enumerate(chunks):
                ep_expert_mapping[ep_id] = chunk # ep_id -> [expert_ids]
                for idx, ele in enumerate(chunk):
                    expert_local_mapping[ele] = idx # expert_id -> local_ep_id
                    expert_ep_mapping[ele] = ep_id # expert_id -> ep_id
            print(f"expert_local_mapping: {expert_local_mapping}")
            print(f"expert_ep_mapping: {expert_ep_mapping}")
            print(f"ep_expert_mapping: {ep_expert_mapping}")

            etp_to_tp_mapping = None
            if etp is not None:
                assert tp % (etp * ep) == 0 or (etp * ep) % tp == 0, f"tp: {tp}, etp: {etp}, ep: {ep}, tp % (etp * ep) != 0"
                etp_to_tp_mapping = {}
                v_tp = tp
                if tp < etp * ep:
                    v_tp = etp * ep
                for t in range(v_tp):
                    etp_id = t % etp
                    ep_id = (t // etp) % ep
                    if ep_id not in etp_to_tp_mapping:
                        etp_to_tp_mapping[ep_id] = {}
                    etp_to_tp_mapping[ep_id][etp_id] = t % tp
            print(f"{etp_to_tp_mapping=}")

        def _convert(layer_name, state_dict_node, p, ep_id, tp_transpose_shape=None, sub_key=None,
                     one_layer_weights_map=None, etp_transpose_shape=None):
            if layer_name not in name_map:
                print(f"> p: {p}. {layer_name}: Not found.")
                return
            no_extra = False
            if sub_key is not None:
                no_extra = False if "extra" in name_map[layer_name][sub_key] and \
                    name_map[layer_name][sub_key]["extra"] else True

            _get_weight_func = {
                INPUT_LAYERNORM: c_ckpt.get_layer_input_layernorm_weight,
                ATTENTION_QUERY_KEY_VALUE: c_ckpt.get_layer_attention_query_key_value_weight,
                ATTENTION_QKV_MAP: c_ckpt.get_layer_attention_weight_by_name,
                ATTENTION_DENSE: c_ckpt.get_layer_attention_dense_weight,
                POST_ATTENTION_LAYERNORM: c_ckpt.get_layer_post_attention_layernorm_weight,
                MOE_GATE: c_ckpt.get_layer_moe_gate_weight,
                MLP_DENSE_H_TO_4H: c_ckpt.get_layer_mlp_dense_h_to_4h_weight,
                MLP_DENSE_4H_TO_H: c_ckpt.get_layer_mlp_dense_4h_to_h_weight,
                POST_MLP_LAYERNORM: c_ckpt.get_layer_post_mlp_layernorm_weight,
            }[layer_name]
            _weight_bias_dict = {
                MOE_GATE: MOE_GATE_BIAS
            }
            _get_bias_func = {
                INPUT_LAYERNORM: c_ckpt.get_layer_input_layernorm_bias,
                ATTENTION_QUERY_KEY_VALUE: c_ckpt.get_layer_attention_query_key_value_bias,
                ATTENTION_QKV_MAP: c_ckpt.get_layer_attention_bias_by_name,
                ATTENTION_DENSE: c_ckpt.get_layer_attention_dense_bias,
                POST_ATTENTION_LAYERNORM: c_ckpt.get_layer_post_attention_layernorm_bias,
                MOE_GATE: c_ckpt.get_layer_moe_gate_bias,
                MLP_DENSE_H_TO_4H: c_ckpt.get_layer_mlp_dense_h_to_4h_bias,
                MLP_DENSE_4H_TO_H: c_ckpt.get_layer_mlp_dense_4h_to_h_bias,
                POST_MLP_LAYERNORM: c_ckpt.get_layer_post_mlp_layernorm_bias,
            }[layer_name]
            _get_layer_norm_weight_func = {
                INPUT_LAYERNORM: lambda x, one_layer_weights=None: None,
                ATTENTION_QKV_MAP: lambda x, one_layer_weights=None: None,
                ATTENTION_DENSE: lambda x, one_layer_weights=None: None,
                POST_ATTENTION_LAYERNORM: lambda x, one_layer_weights=None: None,
                MOE_GATE: lambda x, one_layer_weights=None: None,
                MLP_DENSE_4H_TO_H: lambda x, one_layer_weights=None: None,
                ATTENTION_QUERY_KEY_VALUE: lambda x, one_layer_weights=None: None,
                MLP_DENSE_H_TO_4H: lambda x, one_layer_weights=None: None,
                POST_MLP_LAYERNORM: lambda x, one_layer_weights=None: None,
            }[layer_name]
            _get_layer_norm_bias_func = {
                INPUT_LAYERNORM: lambda x, one_layer_weights=None: None,
                ATTENTION_QKV_MAP: lambda x, one_layer_weights=None: None,
                ATTENTION_DENSE: lambda x, one_layer_weights=None: None,
                POST_ATTENTION_LAYERNORM: lambda x, one_layer_weights=None: None,
                MOE_GATE: lambda x, one_layer_weights=None: None,
                MLP_DENSE_4H_TO_H: lambda x, one_layer_weights=None: None,
                ATTENTION_QUERY_KEY_VALUE: lambda x, one_layer_weights=None: None,
                MLP_DENSE_H_TO_4H: lambda x, one_layer_weights=None: None,
                POST_MLP_LAYERNORM: lambda x, one_layer_weights=None: None,
            }[layer_name]
            a = io.BytesIO()
            torch.save(None, a)
            _get_layer_extra_state_func = {
                INPUT_LAYERNORM: lambda x: None,
                ATTENTION_QKV_MAP: lambda x: None if no_extra else a,
                ATTENTION_DENSE: lambda x: a,
                POST_ATTENTION_LAYERNORM: lambda x: None,
                MLP_DENSE_4H_TO_H: lambda x: a,
                ATTENTION_QUERY_KEY_VALUE: lambda x: a,
                MLP_DENSE_H_TO_4H: lambda x: a,
                POST_MLP_LAYERNORM: lambda x: None,
                MOE_GATE: lambda x: None,
            }[layer_name]
            # input_layernorm
            if not args.no_te:
                if layer_name == INPUT_LAYERNORM:
                    _get_weight_func = lambda x, one_layer_weights=None: None
                    _get_bias_func = lambda x, one_layer_weights=None: None
                if layer_name == ATTENTION_QUERY_KEY_VALUE:
                    _get_layer_norm_weight_func = c_ckpt.get_layer_input_layernorm_weight
                    _get_layer_norm_bias_func = c_ckpt.get_layer_input_layernorm_bias
            # post_attention_layernorm
            if (not args.no_te) and ep is None:
                if layer_name == POST_ATTENTION_LAYERNORM:
                    _get_weight_func = lambda x, one_layer_weights=None: None
                    _get_bias_func = lambda x, one_layer_weights=None: None
                if layer_name == MLP_DENSE_H_TO_4H:
                    _get_layer_norm_weight_func = c_ckpt.get_layer_post_attention_layernorm_weight
                    _get_layer_norm_bias_func = c_ckpt.get_layer_post_attention_layernorm_bias

            weight_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.weight")
            bias_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.bias")
            if sub_key is not None and \
                ("dim" not in name_map[layer_name][sub_key] or not name_map[layer_name][sub_key]["dim"]):
                weight_chunk_dim = None
                bias_chunk_dim = None

            for stage_index in range(stage):
                virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                    dualpipev, stage_index, p, pp, num_layers_in_rank, m_ckpt)
                if layer_for_test is not None:
                    cur_new_layer_index = 0
                for layer_index in range(num_layers_in_rank[virtual_p]):
                    layer_id = layer_index + layer_offset
                    new_layer_index = layer_index
                    if layer_for_test is not None:
                        if layer_id in layer_for_test_map:
                            new_layer_index = cur_new_layer_index
                            cur_new_layer_index += 1
                        else:
                            continue
                    if min_mtp_layer_id is not None and layer_id >= min_mtp_layer_id:
                        cur_layer_prefix = mtp_layer_prefix
                        new_layer_index = layer_id - min_mtp_layer_id
                    else:
                        cur_layer_prefix = layer_prefix
                    one_layer_weights = one_layer_weights_map[layer_id] if one_layer_weights_map is not None else None

                    def update_tensor_per_node(state_dict_node, sub_name, transpose_shape = None, weight_chunk_dim=None, \
                                               is_moe_mlp = False, expert_id = None, is_shared = False):
                        weight_name = None
                        bias_name = None
                        weight_scale_inv_name = None
                        weight_scale_inv = None
                        etp_to_tp = None
                        # weight
                        if is_moe_mlp:
                            name = f"{cur_layer_prefix}.{new_layer_index}.{moe_mlp}.{sub_name}"
                            weight = _get_weight_func(layer_id, is_moe_mlp=True, one_layer_weights=one_layer_weights)
                        elif is_shared:
                            name = f"{cur_layer_prefix}.{new_layer_index}.{moe_shared_expert}.{sub_name}"
                            weight = _get_weight_func(layer_id, is_shared=True, one_layer_weights=one_layer_weights)
                        elif expert_id is not None:
                            local_eid = expert_local_mapping[expert_id]
                            if args.moe_grouped_gemm:
                                name = f"{cur_layer_prefix}.{new_layer_index}.{moe_grouped_gemm_expert}.{sub_name}"
                                weight_name = f"{name}.weight{local_eid}"
                                bias_name = f"{name}.bias{local_eid}"
                            else:
                                name = f"{cur_layer_prefix}.{new_layer_index}.{moe_expert}.{local_eid}.{sub_name}"
                            weight = _get_weight_func(layer_id, expert_id=expert_id, one_layer_weights=one_layer_weights)
                            etp_to_tp = etp_to_tp_mapping[ep_id] if etp is not None else None
                        else:
                            name = f"{cur_layer_prefix}.{new_layer_index}.{sub_name}"
                            if sub_key is not None:
                                if layer_name == ATTENTION_QKV_MAP and \
                                    ("is_layernorm" not in name_map[layer_name][sub_key] or \
                                     not name_map[layer_name][sub_key]["is_layernorm"]):
                                    weight = _get_weight_func(sub_key, layer_id, one_layer_weights=one_layer_weights)
                                else:
                                    weight = _get_weight_func(sub_key, layer_id, one_layer_weights=one_layer_weights)
                                if "is_layernorm" in name_map[layer_name][sub_key] and name_map[layer_name][sub_key]["is_layernorm"]:
                                    weight_name = name + ".layer_norm_weight"
                            else:
                                if layer_name == ATTENTION_DENSE:
                                    weight = _get_weight_func(layer_id, one_layer_weights=one_layer_weights)
                                else:
                                    weight = _get_weight_func(layer_id, one_layer_weights=one_layer_weights)
                            if layer_name in _weight_bias_dict and _weight_bias_dict[layer_name] in name_map:
                                bias_name = f"{cur_layer_prefix}.{new_layer_index}.{name_map[_weight_bias_dict[layer_name]]}"
                            if layer_name == MOE_GATE and num_experts_for_test is not None:
                                weight = weight[:num_experts_for_test]
                        if weight is not None and layer_name in [ATTENTION_QKV_MAP, ATTENTION_DENSE, \
                                                                 MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H]:
                            weight, weight_scale_inv = weight
                        if weight_name is not None:
                            if weight_scale_inv is not None or pretrain_as_fp8:
                                for key in ignore_tp_keys:
                                    if key in weight_name:
                                        weight_chunk_dim = None
                            m_ckpt.update_tensor(state_dict_node, weight, transformer, weight_name,
                                                 dim=weight_chunk_dim, etp_to_tp=etp_to_tp,
                                                 weight_scale_inv=weight_scale_inv, transpose_shape=transpose_shape)
                        else:
                            if weight_scale_inv is not None or pretrain_as_fp8:
                                for key in ignore_tp_keys:
                                    if key in name + ".weight":
                                        weight_chunk_dim = None
                            m_ckpt.update_tensor(state_dict_node, weight, transformer, name + ".weight",
                                                 dim=weight_chunk_dim, etp_to_tp=etp_to_tp,
                                                 weight_scale_inv=weight_scale_inv, transpose_shape=transpose_shape)
                        # bias
                        if sub_key is not None:
                            bias = _get_bias_func(sub_key, layer_id, one_layer_weights=one_layer_weights)
                        else:
                            bias = _get_bias_func(layer_id, one_layer_weights=one_layer_weights)
                        if layer_name == MOE_GATE and num_experts_for_test is not None:
                            bias = bias[:num_experts_for_test]
                        if transpose_shape is not None and bias is not None:
                            bias = transpose_shape0(bias, *transpose_shape)
                        if separate_dtype is not None and layer_name in separate_dtype:
                            bias_type = bias.dtype
                        else:
                            bias_type = None
                        if bias_name is not None:
                            m_ckpt.update_tensor(state_dict_node, bias, transformer, bias_name,
                                                 bias_chunk_dim, dtype=bias_type, etp_to_tp=etp_to_tp)
                        else:
                            m_ckpt.update_tensor(state_dict_node, bias, transformer, name + ".bias",
                                                 bias_chunk_dim, dtype=bias_type, etp_to_tp=etp_to_tp)
                        # layer norm
                        weight = _get_layer_norm_weight_func(layer_id, one_layer_weights=one_layer_weights)
                        bias = _get_layer_norm_bias_func(layer_id, one_layer_weights=one_layer_weights)
                        if transpose_shape is not None:
                            pass # todo
                        m_ckpt.update_tensor(state_dict_node, weight, transformer, name + ".layer_norm_weight",
                                             etp_to_tp=etp_to_tp)
                        m_ckpt.update_tensor(state_dict_node, bias, transformer, name + ".layer_norm_bias",
                                             etp_to_tp=etp_to_tp)
                        # _extra_state
                        if expert_id is None or layer_name not in [MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H] or \
                                (args.moe_grouped_gemm and local_eid == 0):
                            state = _get_layer_extra_state_func(layer_id)
                            m_ckpt.update_tensor(state_dict_node, state, transformer, name + "._extra_state", None)

                    if ep_id is not None and layer_name in [MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H]:
                        if first_k_dense_replace is not None and layer_id < first_k_dense_replace:
                            update_tensor_per_node(state_dict_node, name_map[layer_name], is_moe_mlp=True,
                                                   transpose_shape=tp_transpose_shape,
                                                   weight_chunk_dim=weight_chunk_dim)
                        else:
                            for expert_id in ep_expert_mapping[ep_id]:
                                update_tensor_per_node(state_dict_node, name_map[layer_name], expert_id=expert_id,
                                                       transpose_shape=etp_transpose_shape,
                                                       weight_chunk_dim=weight_chunk_dim)
                            if MOE_SHARED_EXPERT in name_map:
                                update_tensor_per_node(state_dict_node, name_map[layer_name], is_shared=True,
                                                       transpose_shape=tp_transpose_shape,
                                                       weight_chunk_dim=weight_chunk_dim)
                    elif layer_name == MOE_GATE:
                        if first_k_dense_replace is None or layer_id >= first_k_dense_replace:
                            update_tensor_per_node(state_dict_node, name_map[layer_name],
                                                   transpose_shape=tp_transpose_shape,
                                                   weight_chunk_dim=weight_chunk_dim)
                    else:
                        if sub_key is not None:
                            update_tensor_per_node(state_dict_node, name_map[layer_name][sub_key]["name"],
                                                   transpose_shape=tp_transpose_shape,
                                                   weight_chunk_dim=weight_chunk_dim)
                        else:
                            update_tensor_per_node(state_dict_node, name_map[layer_name],
                                                   transpose_shape=tp_transpose_shape,
                                                   weight_chunk_dim=weight_chunk_dim)

            if ep_id is None:
                print(f"> p: {p}. {layer_name} chunk weight dim {weight_chunk_dim}, bias dim {bias_chunk_dim}")
            else:
                print(f"> p: {p}, ep_id: {ep_id}. {layer_name} chunk weight dim {weight_chunk_dim}, bias dim {bias_chunk_dim}")

        def convert_one_p(p, ep_id=None, last_p=None, one_layer_weights_map=None):
            if need_check_dones and (p, ep_id) in done_keys:
                print(f"> p: {p}, ep_id: {ep_id} already converted. pass...")
                return

            if cache_path is None:
                one_layer_weights_map = None
            elif last_p is None or last_p != p:
                if one_layer_weights_map is None:
                    one_layer_weights_map = {}
                one_layer_weights_map.clear()
                for stage_index in range(stage):
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, m_ckpt)
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        if layer_for_test is not None:
                            if layer_id in layer_for_test_map:
                                one_layer_weights_map[layer_id] = torch.load(f"{cache_path}/checkpoint_{layer_id}.pt",
                                                                             map_location="cpu", weights_only=False)
                            else:
                                continue
                        else:
                            one_layer_weights_map[layer_id] = torch.load(f"{cache_path}/checkpoint_{layer_id}.pt",
                                                                         map_location="cpu", weights_only=False)
                        print(f"> load cached weights of layer {layer_id}: {cache_path}/checkpoint_{layer_id}.pt")

            state_dict_node = {}
            if etp is None:
                for t in range(tp):
                    state_dict_node[t] = {}
            else:
                for t in etp_to_tp_mapping[ep_id].values():
                    state_dict_node[t] = {}

            if p == 0:
                # 1.1 word embeddings with paddding
                weight = c_ckpt.get_word_embedding()
                if margs.get("add_embedding_padding", False):
                    divisible_by = margs["make_vocab_size_divisible_by"]
                    vocab_size = cargs["vocab_size"]
                    padded_vocab_size = margs.get("pad_vocab_size_to")
                    if padded_vocab_size is None:
                        assert vocab_size == weight.shape[0]
                    weight = add_embedding_padding(weight, divisible_by, vocab_size, tp, padded_vocab_size)

                if WORD_EMBEDDINGS in name_map:
                    name = m_ckpt.get_word_embedding_name()
                    chunk_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS}.weight")
                    transformer = m_ckpt.get_transformer_name(0)
                    m_ckpt.update_tensor(state_dict_node, weight, transformer, name + ".weight", chunk_dim)
                    print(f"> p: {p}. {name} weight {weight.shape}")

                # 1.2 position embedding
                if margs.get("add_position_embedding", False):
                    name = m_ckpt.get_position_embedding_name()
                    weight = c_ckpt.get_word_position_embedding()
                    chunk_dim = tensor_parallel_dim.get(f"{WORD_POSITION_EMBEDDINGS}.weight")
                    m_ckpt.update_tensor(state_dict_node, weight, name, "weight", chunk_dim)
                    print(f"> p: {p}. {name} weight {weight.shape}")

                if margs.get("add_block_position_embedding", False):
                    name =  m_ckpt.get_block_position_embedding_name()
                    weight = c_ckpt.get_word_block_position_embedding()
                    chunk_dim = tensor_parallel_dim.get(f"{WORD_BLOCK_POSITION_EMBEDDINGS}.weight")
                    m_ckpt.update_tensor(state_dict_node, weight, name, "weight", chunk_dim)
                    print(f"> {name} weight {weight.shape}")

            # 2.1 input_layernorm
            _convert(INPUT_LAYERNORM, state_dict_node, p=p, ep_id=ep_id, one_layer_weights_map=one_layer_weights_map)

            # 2.2 rotary_emb.inv_freqs
            if ATTENTION_ROTARY_EMB_INV_FREQ in name_map:
                chunk_dim = tensor_parallel_dim.get(ATTENTION_ROTARY_EMB_INV_FREQ)
                for stage_index in range(stage):
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, m_ckpt)
                    if layer_for_test is not None:
                        cur_new_layer_index = 0
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        new_layer_index = layer_index
                        if layer_for_test is not None:
                            if layer_id in layer_for_test_map:
                                new_layer_index = cur_new_layer_index
                                cur_new_layer_index += 1
                            else:
                                continue
                        if min_mtp_layer_id is not None and layer_id >= min_mtp_layer_id:
                            cur_layer_prefix = mtp_layer_prefix
                        else:
                            cur_layer_prefix = layer_prefix
                        name = f"{cur_layer_prefix}.{new_layer_index}.{name_map[ATTENTION_ROTARY_EMB_INV_FREQ]}"
                        one_layer_weights = one_layer_weights_map[layer_id] \
                            if one_layer_weights_map is not None else None
                        inv_freq = c_ckpt.get_layer_attention_rotary_emb_inv_freq(
                            layer_id, one_layer_weights=one_layer_weights)
                        m_ckpt.update_tensor(state_dict_node, inv_freq, transformer, name, chunk_dim)
                print(f"> p: {p}. rotary_emb.inv_freqs chunk dim {chunk_dim}")

            # 2.3 self attention query_key_value
            if ATTENTION_QUERY_KEY_VALUE in name_map:
                _convert(ATTENTION_QUERY_KEY_VALUE, state_dict_node, p=p, ep_id=ep_id,
                         one_layer_weights_map=one_layer_weights_map)
            if ATTENTION_QKV_MAP in name_map:
                for sub_key, dict_mcore in name_map[ATTENTION_QKV_MAP].items():
                    _convert(ATTENTION_QKV_MAP, state_dict_node, p=p, ep_id=ep_id, sub_key=sub_key,
                             one_layer_weights_map=one_layer_weights_map)

            # 2.4 self attention dense
            _convert(ATTENTION_DENSE, state_dict_node, p=p, ep_id=ep_id, one_layer_weights_map=one_layer_weights_map)

            # 2.5 post attention layernorm
            _convert(POST_ATTENTION_LAYERNORM, state_dict_node, p=p, ep_id=ep_id, one_layer_weights_map=one_layer_weights_map)

            # 2.6 mlp dense h_to_4h
            if ep is not None:
                _convert(MOE_GATE, state_dict_node, p=p, ep_id=ep_id, one_layer_weights_map=one_layer_weights_map)
            if margs.get("transpose_mlp_dense", False):
                if etp is None:
                    etp_transpose_shape = (2, tp)
                else:
                    etp_transpose_shape = (2, etp)
                _convert(MLP_DENSE_H_TO_4H, state_dict_node, p=p, ep_id=ep_id, tp_transpose_shape=(2, tp),
                         etp_transpose_shape=etp_transpose_shape, one_layer_weights_map=one_layer_weights_map)
            else:
                _convert(MLP_DENSE_H_TO_4H, state_dict_node, p=p, ep_id=ep_id,
                         one_layer_weights_map=one_layer_weights_map)

            # 2.7 mlp dense 4h_to_h
            _convert(MLP_DENSE_4H_TO_H, state_dict_node, p=p, ep_id=ep_id, one_layer_weights_map=one_layer_weights_map)

            # 2.8 post mlp layernorm
            _convert(POST_MLP_LAYERNORM, state_dict_node, p=p, ep_id=ep_id, one_layer_weights_map=one_layer_weights_map)
            add_final_layer = False
            if not dualpipev and p == pp - 1:
                add_final_layer = True
            elif dualpipev and p == 0:
                add_final_layer = True
            if add_final_layer:
                # 2.9 final_layernorm
                if FINAL_LAYERNORM in name_map:
                    transformer = m_ckpt.get_transformer_name(m_ckpt.num_stages - 1)
                    name = name_map[FINAL_LAYERNORM]
                    # weight
                    chunk_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.weight")
                    weight = c_ckpt.get_final_layernorm_weight()
                    m_ckpt.update_tensor(state_dict_node, weight, transformer, name + ".weight", chunk_dim)
                    # bias
                    chunk_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.bias")
                    bias = c_ckpt.get_final_layernorm_bias()
                    m_ckpt.update_tensor(state_dict_node, bias, transformer, name + ".bias", chunk_dim)
                    print(f"> p: {p}. {name} weight {weight.shape}")

                # 3 word embedding for head
                if WORD_EMBEDDINGS_FOR_HEAD in name_map:
                    if margs.get("untie_embeddings_and_output_weights", False) or m_ckpt.pp > 1:
                        chunk_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS_FOR_HEAD}.weight")
                        name = m_ckpt.get_word_embedding_for_head_name()
                        weight = c_ckpt.get_word_embeddings_for_head_weight()
                        if margs.get("add_embedding_padding", False):
                            divisible_by = margs["make_vocab_size_divisible_by"]
                            orig_vocab_size = cargs["vocab_size"]
                            padded_vocab_size = margs.get("pad_vocab_size_to")
                            weight = add_embedding_padding(weight, divisible_by, orig_vocab_size, tp, padded_vocab_size)
                        m_ckpt.update_tensor(state_dict_node, weight, transformer, name + ".weight", chunk_dim)
                        print(f"> p: {p}. {name} weight {weight.shape}")

                if MTP_WORD_EMBEDDING in name_map:
                    virtual_p = pp * m_ckpt.num_stages - 1
                    layer_offset = sum(num_layers_in_rank[:virtual_p])
                    transformer = m_ckpt.get_transformer_name(m_ckpt.num_stages - 1)
                    new_layer_index = 0
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        if layer_id < ori_num_layers:
                            continue
                        if layer_for_test is not None and layer_id not in layer_for_test_map:
                            continue

                        one_layer_weights = one_layer_weights_map[layer_id] \
                            if one_layer_weights_map is not None else None
                        (mtp_word_embedding, mtp_enorm, mtp_hnorm, mtp_eh_proj), \
                            (mtp_shared_head_norm, mtp_shared_head_head) = c_ckpt.get_layer_mtp_weight(
                                layer_id, one_layer_weights=one_layer_weights)
                        if margs.get("add_embedding_padding", False):
                            divisible_by = margs["make_vocab_size_divisible_by"]
                            vocab_size = cargs["vocab_size"]
                            padded_vocab_size = margs.get("pad_vocab_size_to")
                            if padded_vocab_size is None:
                                assert vocab_size == mtp_word_embedding.shape[0]
                            mtp_word_embedding = add_embedding_padding(mtp_word_embedding, divisible_by,
                                                                       vocab_size, tp, padded_vocab_size)

                        name = f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_WORD_EMBEDDING]}.weight"
                        m_ckpt.update_tensor(state_dict_node, mtp_word_embedding, transformer, \
                                            name, tensor_parallel_dim.get(f"{MTP_WORD_EMBEDDING}.weight"))
                        name = f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_ENORM]}.weight"
                        m_ckpt.update_tensor(state_dict_node, mtp_enorm, transformer, \
                                            name, tensor_parallel_dim.get(f"{MTP_ENORM}.weight"))
                        name = f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_HNORM]}.weight"
                        m_ckpt.update_tensor(state_dict_node, mtp_hnorm, transformer, \
                                            name, tensor_parallel_dim.get(f"{MTP_HNORM}.weight"))
                        name = f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_EH_PROJ]}.weight"
                        m_ckpt.update_tensor(state_dict_node, mtp_eh_proj, transformer, \
                                            name, tensor_parallel_dim.get(f"{MTP_EH_PROJ}.weight"))
                        name = f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_SHARED_HEAD_NORM]}.weight"
                        m_ckpt.update_tensor(state_dict_node, mtp_shared_head_norm, transformer, \
                                            name, tensor_parallel_dim.get(f"{MTP_SHARED_HEAD_NORM}.weight"))
                        #name = f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_SHARED_HEAD_HEAD]}.weight"
                        # m_ckpt.update_tensor(state_dict_node, mtp_shared_head_head, transformer, \
                        #                     name, tensor_parallel_dim.get(f"{MTP_SHARED_HEAD_HEAD}.weight"))
                        print(f"> p: {p}. layer_index: {new_layer_index}, "\
                                f"mtp_word_embedding: {mtp_word_embedding.shape}, "\
                                f"mtp_enorm: {mtp_enorm.shape}, mtp_hnorm: {mtp_hnorm.shape}, "\
                                f"mtp_eh_proj: {mtp_eh_proj.shape}, mtp_shared_head_norm: {mtp_shared_head_norm.shape}")
                        new_layer_index += 1

            saved_models_str = {}
            for stage_index in range(stage):
                saved_layer_ids = []
                virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                    dualpipev, stage_index, p, pp, num_layers_in_rank, m_ckpt)
                for layer_index in range(num_layers_in_rank[virtual_p]):
                    layer_id = layer_index + layer_offset
                    if layer_for_test is not None:
                        if layer_id in layer_for_test_map:
                            saved_layer_ids.append(str(layer_id))
                    else:
                        saved_layer_ids.append(str(layer_id))
                saved_models_str[transformer] = ", ".join(saved_layer_ids)

            for t in state_dict_node.keys():
                if ep_id is None:
                    m_ckpt.save_model_file(
                        release_dir, save_margs, p, t, None, state_dict_node[t],
                        m_ckpt.optim_state_dict[p][t] if m_ckpt.optim_state_dict is not None else None, saved_models_str)
                else:
                    m_ckpt.save_model_file(
                        release_dir, save_margs, p, t, ep_id, state_dict_node[t],
                        m_ckpt.optim_state_dict[p][ep_id][t] if m_ckpt.optim_state_dict is not None else None,
                        saved_models_str)

            # optimizer
            if use_distributed_optimizer and save_optim:
                for t in range(tp):
                    if ep_id is None:
                        m_ckpt.save_optimizer(release_dir, p, t, None, state_dict_node[t])
                    else:
                        m_ckpt.save_optimizer(release_dir, p, t, ep_id, state_dict_node[t][ep_id])
            touch_file(done_dir=done_dir, p_id=p, ep_id=ep_id)

        if cache_path is None:
            one_layer_weights_map = None
        else:
            one_layer_weights_map = {}
        if args.max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                for p in range(pp):
                    if ep is None:
                        futures.append(executor.submit(convert_one_p, p=p, ep_id=None,
                                                       one_layer_weights_map=one_layer_weights_map))
                    else:
                        for ep_id in range(ep):
                            futures.append(executor.submit(convert_one_p, p=p, ep_id=ep_id,
                                                           one_layer_weights_map=one_layer_weights_map))

            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise e
        else:
            last_p = None
            for p in range(pp):
                if ep is None:
                    convert_one_p(p=p, ep_id=None, last_p=last_p, one_layer_weights_map=one_layer_weights_map)
                    last_p = p
                else:
                    for ep_id in range(ep):
                        convert_one_p(p=p, ep_id=ep_id, last_p=last_p, one_layer_weights_map=one_layer_weights_map)
                        last_p = p
        checked_done = check_all_done(done_dir, pp, ep)
        assert checked_done, f"{done_dir} is not complete. please retry it again"
        shutil.rmtree(done_dir)

        m_ckpt.debug("Finish common -> mcore")
        return m_ckpt

    def update_tensor(self, state_dict_node, source, layer, key, dim=None, dtype=None, etp_to_tp=None,
                      weight_scale_inv=None, transpose_shape = None):
        """
        Update a tensor, which can be a torch.Tensor or other types of objects. If it is a torch.Tensor,
        it will be split into multiple parts and updated; otherwise, it will be updated directly.
        If dim is not None, the source tensor is divided into multiple parts,
        and each part is updated to the corresponding layer and key in state_dict_node.

        Args:
            state_dict_node (List[Dict]): A list containing multiple elements,
                each of which is a dictionary used to store model parameters.
            Each element is a dictionary that stores model parameters in the format
                {layer1: {param1: value1}, layer2: {param2: value2}}.
            source (Union[torch.Tensor, Any]): The source tensor or other types of objects that need to be updated.
                If it is a torch.Tensor, it will be split into multiple parts and updated; otherwise,
                it will be updated directly.
            layer (str): The name of the layer to be updated.
            key (str): The key name that needs to be updated.
            dim (Optional[int], optional): If the source tensor is 3D or higher,
                the dimensions of the split need to be specified. The default is None,
                which means no split is performed. Defaults to None

        Returns:
            None.
        """
        dtype = self.dtype if dtype is None else dtype
        args = parse_args()

        if args.fp8_quant_transfer_type == "bfloat16":
            transfer_dtype = torch.bfloat16
        else:
            transfer_dtype = torch.float32

        def get_quantizer_with_weight_scale_inv(weight, weight_scale_inv):
            from transformer_engine.pytorch.tensor.float8_blockwise_tensor \
                import Float8BlockwiseQTensor
            from transformer_engine.pytorch.constants import TE_DType
            return Float8BlockwiseQTensor(
                rowwise_data=weight,
                rowwise_scale_inv=weight_scale_inv,
                columnwise_data=None,
                columnwise_scale_inv=None,
                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                quantizer=None,
                is_2D_scaled=True,
                shape=weight.shape,
                dtype=torch.bfloat16
            )

        def chunk_fp8_weight(weight, weight_scale_inv, cur_tp, transpose_shape=None):
            weight_bf16 = McoreCheckpoint.per_block_dequant_from_fp8(weight, weight_scale_inv, dtype=transfer_dtype)
            if transpose_shape is not None:
                weight_bf16 = transpose_shape0(weight_bf16, *transpose_shape)
            weight_bf16_s = torch.chunk(weight_bf16, cur_tp, dim=dim)
            weight_s = []
            weight_scale_inv_s = []
            for w_bf16 in weight_bf16_s:
                weight, weight_scale_inv = McoreCheckpoint.per_block_cast_to_fp8(
                    w_bf16, method=args.quant_method, amax_epsilon=args.amax_epsilon,
                    force_pow_2_scales=args.force_pow_2_scales)
                weight_s.append(weight)
                weight_scale_inv_s.append(weight_scale_inv)
            return weight_s, weight_scale_inv_s

        if isinstance(source, torch.Tensor):
            shape = source.shape
            tp_shapes = []
            if etp_to_tp is None:
                cur_tp = self.tp
            else:
                cur_tp = self.etp
            need_chunk = dim is not None and cur_tp > 1
            if need_chunk:
                if weight_scale_inv is None:
                    if transpose_shape is not None:
                        source = transpose_shape0(source, *transpose_shape)
                    source = torch.chunk(source, cur_tp, dim=dim)
                    for s in source:
                        tp_shapes.append(s.shape)
                else:
                    source, weight_scale_inv = chunk_fp8_weight(
                        source, weight_scale_inv, cur_tp, transpose_shape=transpose_shape)
                    for s in source:
                        tp_shapes.append(s.shape)
                    for inv in weight_scale_inv:
                        tp_shapes.append(inv.shape)
            elif weight_scale_inv is not None:
                weight_bf16 = McoreCheckpoint.per_block_dequant_from_fp8(source, weight_scale_inv, dtype=transfer_dtype)
                source, weight_scale_inv = McoreCheckpoint.per_block_cast_to_fp8(
                    weight_bf16, method=args.quant_method, amax_epsilon=args.amax_epsilon,
                    force_pow_2_scales=args.force_pow_2_scales)

            if etp_to_tp is None:
                for t in state_dict_node.keys():
                    element = get_element_from_dict_by_path(
                        state_dict_node[t], layer
                    )
                    if weight_scale_inv is None:
                        element[key] = (source if not need_chunk else source[t].clone())
                    else:
                        element[key] = get_quantizer_with_weight_scale_inv(
                            weight=(source if not need_chunk else source[t].clone()),
                            weight_scale_inv=(weight_scale_inv if not need_chunk else weight_scale_inv[t].clone())
                        )
            else:
                for et in range(self.etp):
                    t = etp_to_tp[et]
                    element = get_element_from_dict_by_path(
                        state_dict_node[t], layer
                    )
                    if weight_scale_inv is None:
                        element[key] = (source if not need_chunk else source[et].clone())
                    else:
                        element[key] = get_quantizer_with_weight_scale_inv(
                            weight = (source if not need_chunk else source[et].clone()),
                            weight_scale_inv=(weight_scale_inv if not need_chunk else weight_scale_inv[et].clone()),
                        )

            print(f"update_tensor: {key}, {shape=}, {tp_shapes=}")
        elif source is not None:
            for t in state_dict_node.keys():
                element = get_element_from_dict_by_path(
                    state_dict_node[t], layer
                )
                element[key] = source
            print(f"update_tensor: {key}")

    def is_fp8_q_tensor(self, state_dict, layer, key):
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor \
            import Float8BlockwiseQTensor
        element = get_element_from_dict_by_path(
            state_dict[0], layer
        )
        if key not in element:
            return False
        return isinstance(element[key], Float8BlockwiseQTensor)

    def get_tensor(self, state_dict, layer, key, dim=None, etp_to_tp=None, transpose_shape=None):
        args = parse_args()
        if args.pretrain_as_fp8:
            from transformer_engine.pytorch.tensor.float8_blockwise_tensor \
                import Float8BlockwiseQTensor

        def cat_fp8_weight(tp_weight_dict, tp_weight_scale_inv, transpose_shape=None):
            weight_bf16_s = []
            if args.fp8_quant_transfer_type == "bfloat16":
                transfer_dtype = torch.bfloat16
            else:
                transfer_dtype = torch.float32
            for i in range(len(tp_weight_dict)):
                weight = tp_weight_dict[i]
                weight_scale_inv = tp_weight_scale_inv[i]
                weight_bf16 = McoreCheckpoint.per_block_dequant_from_fp8(weight, weight_scale_inv, dtype=transfer_dtype)
                weight_bf16_s.append(weight_bf16)
            weight_bf16 = torch.cat(weight_bf16_s, dim=dim)

            if transpose_shape is not None:
                weight_bf16 = transpose_shape0(weight_bf16, *transpose_shape)

            weight, weight_scale_inv = McoreCheckpoint.per_block_cast_to_fp8(
                weight_bf16, method=args.quant_method, amax_epsilon=args.amax_epsilon,
                force_pow_2_scales=args.force_pow_2_scales)

            return weight, weight_scale_inv

        def get_weight_dict(cur_tp):
            tp_weight_dict = []
            tp_weight_scale_inv = []
            for t in range(cur_tp):
                element = get_element_from_dict_by_path(
                    state_dict[t], layer
                )
                if key not in element:
                    return None, None
                if args.pretrain_as_fp8 and isinstance(element[key], Float8BlockwiseQTensor):
                    tp_weight_dict.append(element[key]._rowwise_data)
                    tp_weight_scale_inv.append(element[key]._rowwise_scale_inv)
                else:
                    tp_weight_dict.append(element[key])
            return tp_weight_dict, tp_weight_scale_inv

        if dim is not None:
            if etp_to_tp is None:
                cur_tp = self.tp
            else:
                cur_tp = self.etp
            tp_weight_dict, tp_weight_scale_inv = get_weight_dict(cur_tp)
            if tp_weight_dict is None:
                return None, None

            if tp_weight_scale_inv is None or len(tp_weight_scale_inv) == 0:
                source = torch.cat(tp_weight_dict, dim=dim)
                weight_scale_inv = None
                if transpose_shape is not None:
                    source = transpose_shape0(source, *transpose_shape)
            else:
                weight_len = 0
                scale_inv_len = 0
                for weight in tp_weight_dict:
                    weight_len += weight.shape[0]
                for scale_inv in tp_weight_scale_inv:
                    scale_inv_len += scale_inv.shape[0]
                source, weight_scale_inv = cat_fp8_weight(
                    tp_weight_dict, tp_weight_scale_inv, transpose_shape=transpose_shape)
        else:
            element = get_element_from_dict_by_path(
                state_dict[0], layer
            )
            if key not in element:
                return None, None
            if args.pretrain_as_fp8 and isinstance(element[key], Float8BlockwiseQTensor):
                source = element[key]._rowwise_data
                weight_scale_inv = element[key]._rowwise_scale_inv
            else:
                source = element[key]
                weight_scale_inv = None
        if source is not None:
            print(f"get_tensor: {key}, {source.shape}")
        return source, weight_scale_inv

    @staticmethod
    def per_block_dequant_from_fp8(fp8_blocks: torch.Tensor,
                                scales: torch.Tensor,
                                orig_shape: Tuple[int, int] | None = None,
                                dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        m_full, n_full = fp8_blocks.shape
        if orig_shape is None:
            m, n = m_full, n_full
        else:
            m, n = orig_shape
            # sanity check
            if m > m_full or n > n_full:
                raise ValueError("orig_shape is larger than fp8 tensor shape")

        m_pad = ceil_div(m, 128) * 128
        n_pad = ceil_div(n, 128) * 128

        if m_pad == m_full and n_pad == n_full:
            fp8_padded = fp8_blocks
        else:
            fp8_padded = torch.zeros((m_pad, n_pad), dtype=fp8_blocks.dtype, device=fp8_blocks.device)
            fp8_padded[:m, :n] = fp8_blocks

        fp8_view = fp8_padded.view(-1, 128, n_pad // 128, 128)
        scale_expanded = scales.view(-1, 1, scales.size(1), 1)
        x_recon_view = fp8_view.to(scale_expanded.dtype) * scale_expanded
        x_recon = x_recon_view.view(m_pad, n_pad)[:m, :n].contiguous()
        x_recon = x_recon.to(dtype)

        return x_recon

    @staticmethod
    def per_block_cast_to_fp8(
        x: torch.Tensor,
        method: Literal["te", "pt", "aiak"] = 'te',
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 2d tensor, the model parameter what will be block-wise quantize to fp8.
            method: one of the ["te", "pt", "aiak"], means using TransformerEngine, naive PyTorch, aiak-fp8-quantizer
                to do the quantization respectively. Defaults to "te".
            fp8_dtype: the dtype of the output fp8 tensor. Defaults to torch.float8_e4m3fn.
            **kwargs: kwargs pass to the `te` method. Take no effect in other quantization methods. Belows are the args:
                `amax_epsilon`: defaults to 0.
                `force_pow_2_scales` defaults to True.
        Returns:
            x_scaled: 2d tensor, the quantized tensor.
            weight_scale_inv: 2d tensor, the scale_inv of the quantized tensor.
        """

        # Always do the quantization on device
        x = x.cuda()
        
        if method == "pt":
            assert x.dim() == 2
            m, n = x.shape
            x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
            x_padded[:m, :n] = x
            x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
            x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
            x_scaled = (x_view * (448.0 / x_amax)).to(fp8_dtype)
            return x_scaled.view_as(x_padded)[:m, :n].contiguous().cpu(), \
                (x_amax / 448.0).view(x_view.size(0), x_view.size(2)).cpu()

        elif method == "aiak":
            import aiak_fp8_quantizer
            weight, weight_scale_inv, *_ = aiak_fp8_quantizer.per_block_cast_to_fp8_fprop_vector(x)
            return weight.view(fp8_dtype).cpu(), weight_scale_inv.cpu()

        elif method == "te":
            from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
            from transformer_engine.pytorch.constants import TE_DType
            quantizer = Float8BlockQuantizer(
                fp8_dtype=TE_DType[fp8_dtype],
                rowwise=True,
                columnwise=False,
                amax_epsilon=kwargs.get("amax_epsilon", 0.0),
                force_pow_2_scales=kwargs.get("force_pow_2_scales", True),
                block_scaling_dim=2,
            )
            xq = quantizer(x)
            fp8_weight = xq._rowwise_data.view(fp8_dtype).cpu()  # `_rowwise_data` is torch.uint8
            fp8_scale_inv = xq._rowwise_scale_inv.cpu()
            return fp8_weight, fp8_scale_inv
        else:
            raise ValueError(f"invalid quantization method: {method}")

    def convert_to_common(self, c_config):
        """
        Convert Mcore checkpoint to common checkpoint.
            Args:
                c_config: CommonConfig
        """

        print("\n==================== Mcore -> Common ====================")

        tensor_parallel_dim = self.tensor_parallel_dim
        name_map = c_config.get_name_map("mcore")
        hargs = c_config.get_args("huggingface")
        cargs = c_config.get_args("common")
        margs = c_config.get_args("mcore")
        num_nextn_predict_layers = hargs.get("num_nextn_predict_layers", 0)
        ori_num_layers = cargs["num_layers"]
        num_layers = ori_num_layers + num_nextn_predict_layers
        hidden_size = cargs["hidden_size"]
        num_attention_heads = cargs["num_attention_heads"]
        args = parse_args()
        custom_pipeline_layers = args.custom_pipeline_layers
        cache_path = args.cache_path
        if cache_path is not None:
            os.makedirs(cache_path, exist_ok=True)
        tp = args.tensor_model_parallel_size
        ep = args.expert_parallel_size
        pp = args.pipeline_model_parallel_size
        etp = args.expert_tensor_parallel_size if hasattr(args, 'expert_tensor_parallel_size') else None
        dualpipev = args.vpp_scheduler == 'dualpipev'
        num_experts = args.num_experts
        convert_to_fp8 = args.convert_to_fp8
        if convert_to_fp8:
            assert "fp8_keys" in margs, "fp8_keys should be defined when convert_to_fp8 is True"
            fp8_keys = margs["fp8_keys"]
        else:
            fp8_keys = None
        ignore_tp_keys = margs.get("ignore_tp_keys", [])
        pretrain_as_fp8 = args.pretrain_as_fp8
        if num_experts is not None:
            assert num_experts > 0, "num_experts must be greater than zero"
            if ep is None:
                ep = 1  # if ep is not set, will process the model as no MoE

        c_ckpt = CommonCheckpoint(num_layers)
        c_ckpt.set_dtype(self.dtype)

        layer_prefix = name_map[LAYER_PREFIX]

        num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            assert args.num_virtual_stages_per_pipeline_rank is not None, "num_virtual_stages_per_pipeline_rank is required"
        num_layers_per_stage = margs["num_layers_per_virtual_pipeline_stage"]
        if num_layers_per_stage:
            stage = num_layers // pp // num_layers_per_stage
        else:
            stage = args.num_virtual_stages_per_pipeline_rank or 1
        first_k_dense_replace = margs.get("first_k_dense_replace", None)
        if custom_pipeline_layers is not None:
            assert num_layers_in_first_pipeline_stage is None and num_layers_in_last_pipeline_stage is None, \
                "custom_pipeline_layers need not num_layers_in_first_pipeline_stage or in_last_pipeline_stage"
            num_layers_in_rank, _ = custom_partition_imbalanced(
                ori_num_layers, self.pp * stage, custom_pipeline_layers)
            num_layers_in_rank[-1] += num_nextn_predict_layers
        elif num_layers_in_first_pipeline_stage is not None or num_layers_in_last_pipeline_stage is not None:
            num_layers_in_rank = uneven_vpp_partition(num_layers, pp, stage, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage)
            num_layers_in_rank[-1] += num_nextn_predict_layers
        else:
            num_layers_in_rank, _ = partition_balanced(num_layers, self.pp * self.num_stages)

        # get specific layer for test
        layer_for_test = args.layer_for_test
        if layer_for_test is not None:
            splits = []
            if layer_for_test.find(',') != -1:
                splits = [int(s) for s in layer_for_test.split(',')]
            else:
                splits = [int(layer_for_test)]
            layer_for_test_map = {}
            for index, layer_id in enumerate(splits):
                layer_for_test_map[layer_id] = index

        if num_nextn_predict_layers > 0:
            min_mtp_layer_id = ori_num_layers
            mtp_layer_prefix = name_map[MTP_LAYER_PREFIX]
        else:
            min_mtp_layer_id = None

        if ep is not None:
            post_attention_layernorm = name_map[POST_ATTENTION_LAYERNORM]
            if MOE_MLP in name_map:
                moe_mlp = name_map[MOE_MLP]
            moe_expert = name_map[MOE_EXPERT]
            if MOE_GROUPED_GEMM_EXPERT in name_map:
                moe_grouped_gemm_expert = name_map[MOE_GROUPED_GEMM_EXPERT]
            if MOE_SHARED_EXPERT in name_map:
                moe_shared_expert = name_map[MOE_SHARED_EXPERT]

            experts_ids = [x for x in range(num_experts)]
            chunks = [experts_ids[x:x + num_experts // ep]
                for x in range(0, len(experts_ids), num_experts // ep)] # ep_id -> [expert_ids]

            expert_local_mapping = {}
            expert_ep_mapping = {}
            ep_expert_mapping = {}
            for ep_id, chunk in enumerate(chunks):
                ep_expert_mapping[ep_id] = chunk # ep_id -> [expert_ids]
                for idx, ele in enumerate(chunk):
                    expert_local_mapping[ele] = idx # expert_id -> local_ep_id
                    expert_ep_mapping[ele] = ep_id # expert_id -> ep_id
            print(f"expert_local_mapping: {expert_local_mapping}")
            print(f"expert_ep_mapping: {expert_ep_mapping}")
            print(f"ep_expert_mapping: {ep_expert_mapping}")

            etp_to_tp_mapping = None
            if etp is not None:
                assert tp % (etp * ep) == 0 or (etp * ep) % tp == 0, f"tp: {tp}, etp: {etp}, ep: {ep}, tp % (etp * ep) != 0"
                etp_to_tp_mapping = {}
                v_tp = tp
                if tp < etp * ep:
                    v_tp = etp * ep
                for t in range (v_tp):
                    etp_id = t % etp
                    ep_id = (t // etp) % ep
                    if ep_id not in etp_to_tp_mapping:
                        etp_to_tp_mapping[ep_id] = {}
                    etp_to_tp_mapping[ep_id][etp_id] = t % tp
            print(f"{etp_to_tp_mapping=}")

        if args.save_sub_checkpoint_by_pp and utils.LOADED_STATE_DICT is not None:
            utils.LOADED_LAYERS = set()
            if num_experts is not None:
                utils.LOADED_MIN_E = min(e for p in utils.LOADED_STATE_DICT for e in utils.LOADED_STATE_DICT[p])

        def _convert(layer_name, p, state_dict_node, ep_id=None, tp_transpose_shape=None, sub_key=None,
                     expert_state_dict=None, one_layer_weights_map=None, etp_transpose_shape=None):
            if one_layer_weights_map is not None and len(one_layer_weights_map) == 0:
                return
            _set_func = {
                INPUT_LAYERNORM: c_ckpt.set_layer_input_layernorm,
                ATTENTION_QUERY_KEY_VALUE: c_ckpt.set_layer_attention_query_key_value,
                ATTENTION_QKV_MAP: c_ckpt.set_layer_attention_by_name,
                ATTENTION_DENSE: c_ckpt.set_layer_attention_dense,
                POST_ATTENTION_LAYERNORM: c_ckpt.set_layer_post_attention_layernorm,
                MOE_GATE: c_ckpt.set_layer_moe_gate,
                MLP_DENSE_H_TO_4H: c_ckpt.set_layer_mlp_dense_h_to_4h,
                MLP_DENSE_4H_TO_H: c_ckpt.set_layer_mlp_dense_4h_to_h,
                POST_MLP_LAYERNORM: c_ckpt.set_layer_post_mlp_layernorm,
            }[layer_name]
            _weight_bias_dict = {
                MOE_GATE: MOE_GATE_BIAS
            }
            _set_layernorm_func = {
                INPUT_LAYERNORM: lambda x, y, z, one_layer_weights=None: None,
                ATTENTION_QUERY_KEY_VALUE: lambda x, y, z, one_layer_weights=None: None,
                ATTENTION_QKV_MAP: lambda x, y, z, one_layer_weights=None: None,
                ATTENTION_DENSE: lambda x, y, z, one_layer_weights=None: None,
                POST_ATTENTION_LAYERNORM: lambda x, y, z, one_layer_weights=None: None,
                MOE_GATE: lambda x, y, z, one_layer_weights=None: None,
                MLP_DENSE_H_TO_4H: lambda x, y, z, one_layer_weights=None: None,
                MLP_DENSE_4H_TO_H: lambda x, y, z, one_layer_weights=None: None,
                POST_MLP_LAYERNORM: lambda x, y, z, one_layer_weights=None: None,
            }[layer_name]
            # input_layernorm
            if not args.no_te:
                if layer_name == INPUT_LAYERNORM:
                    _set_func = lambda x, y, z, one_layer_weights=None: None
                if layer_name == ATTENTION_QUERY_KEY_VALUE:
                    _set_layernorm_func = c_ckpt.set_layer_input_layernorm
            # post_attention_layernorm
            if (not args.no_te) and ep is None:
                if layer_name == POST_ATTENTION_LAYERNORM:
                    _set_func = lambda x, y, z, one_layer_weights=None: None
                if layer_name == MLP_DENSE_H_TO_4H:
                    _set_layernorm_func = c_ckpt.set_layer_post_attention_layernorm

            weight_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.weight")
            bias_chunk_dim = tensor_parallel_dim.get(f"{layer_name}.bias")
            if sub_key is not None and \
                ("dim" not in name_map[layer_name][sub_key] or not name_map[layer_name][sub_key]["dim"]):
                weight_chunk_dim = None
                bias_chunk_dim = None

            for stage_index in range(stage):
                virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                    dualpipev, stage_index, p, pp, num_layers_in_rank, self)

                if layer_for_test is not None:
                    cur_new_layer_index = 0
                for layer_index in range(num_layers_in_rank[virtual_p]):
                    layer_id = layer_index + layer_offset
                    new_layer_index = layer_index
                    if layer_for_test is not None:
                        if layer_id in layer_for_test_map:
                            new_layer_index = cur_new_layer_index
                            cur_new_layer_index += 1
                        else:
                            continue
                    if min_mtp_layer_id is not None and layer_id >= min_mtp_layer_id:
                        cur_layer_prefix = mtp_layer_prefix
                        new_layer_index = layer_id - min_mtp_layer_id
                    else:
                        cur_layer_prefix = layer_prefix
                    one_layer_weights = one_layer_weights_map[layer_id] \
                        if one_layer_weights_map is not None else None

                    def get_tensor_per_node(transformer, sub_name, is_moe_mlp = False, expert_id = None, is_shared = False,
                                            transpose_shape=None, weight_chunk_dim=None):
                        weight_name = None
                        bias_name = None
                        weight_scale_inv_name = None
                        etp_to_tp = None
                        cur_state_dict = state_dict_node if expert_id is None else expert_state_dict
                        if is_moe_mlp:
                            name = f"{cur_layer_prefix}.{new_layer_index}.{moe_mlp}.{sub_name}"
                        elif is_shared:
                            name = f"{cur_layer_prefix}.{new_layer_index}.{moe_shared_expert}.{sub_name}"
                        elif expert_id is not None:
                            local_eid = expert_local_mapping[expert_id]
                            if args.moe_grouped_gemm:
                                name = f"{cur_layer_prefix}.{new_layer_index}.{moe_grouped_gemm_expert}.{sub_name}"
                                weight_name = f"{name}.weight{local_eid}"
                                bias_name = f"{name}.bias{local_eid}"
                            else:
                                name = f"{cur_layer_prefix}.{new_layer_index}.{moe_expert}.{local_eid}.{sub_name}"
                            etp_to_tp = etp_to_tp_mapping[ep_id] if etp is not None else None
                        else:
                            name = f"{cur_layer_prefix}.{new_layer_index}.{sub_name}"
                            if sub_key is not None and "is_layernorm" in name_map[layer_name][sub_key] and name_map[layer_name][sub_key]["is_layernorm"]:
                                weight_name = name + ".layer_norm_weight"
                            if layer_name in _weight_bias_dict and _weight_bias_dict[layer_name] in name_map:
                                bias_name = f"{cur_layer_prefix}.{new_layer_index}.{name_map[_weight_bias_dict[layer_name]]}"

                        if weight_name is None:
                            weight_name = name + ".weight"
                        if pretrain_as_fp8:
                            for key in ignore_tp_keys:
                                if key in weight_name:
                                    weight_chunk_dim = None
                        weight, weight_scale_inv = self.get_tensor(
                            cur_state_dict, transformer, weight_name, dim=weight_chunk_dim, etp_to_tp=etp_to_tp)
                        need_to_convert_fp8 = False
                        if convert_to_fp8:
                            for fp8_key in fp8_keys:
                                if fp8_key in weight_name:
                                    need_to_convert_fp8 = True

                        if weight is None and utils.LOADED_LAYERS is not None:
                            return
                        if bias_name is not None:
                            bias, _ = self.get_tensor(cur_state_dict, transformer, bias_name, bias_chunk_dim,
                                                   etp_to_tp=etp_to_tp)
                        else:
                            bias, _ = self.get_tensor(cur_state_dict, transformer, name + ".bias", bias_chunk_dim,
                                                   etp_to_tp=etp_to_tp)
                        layernorm_weight, _ = self.get_tensor(cur_state_dict, transformer, name + ".layer_norm_weight",
                                                           etp_to_tp=etp_to_tp)
                        layernorm_bias, _ = self.get_tensor(cur_state_dict, transformer, name + ".layer_norm_bias",
                                                         etp_to_tp=etp_to_tp)
                        if transpose_shape is not None:
                            weight = transpose_shape0(weight, *transpose_shape)
                            if bias is not None:
                                bias = transpose_shape0(bias, *transpose_shape)
                            if weight_scale_inv is not None:
                                weight_scale_inv = transpose_shape0(weight_scale_inv, *transpose_shape)

                        if weight_scale_inv is None and need_to_convert_fp8:
                            weight, weight_scale_inv = McoreCheckpoint.per_block_cast_to_fp8(
                                weight, method=args.quant_method, amax_epsilon=args.amax_epsilon,
                                force_pow_2_scales=args.force_pow_2_scales)
                            weight = weight.cpu()
                            weight_scale_inv = weight_scale_inv.cpu()

                        if layer_name in [MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H]:
                            if is_moe_mlp:
                                _set_func(layer_id, weight, bias, is_moe_mlp = is_moe_mlp,
                                          one_layer_weights=one_layer_weights, weight_scale_inv=weight_scale_inv)
                            elif is_shared:
                                _set_func(layer_id, weight, bias, is_shared = is_shared,
                                          one_layer_weights=one_layer_weights, weight_scale_inv=weight_scale_inv)
                            elif expert_id is not None:
                                _set_func(layer_id, weight, bias, expert_id = expert_id,
                                          one_layer_weights=one_layer_weights, weight_scale_inv=weight_scale_inv)
                            else:
                                _set_func(layer_id, weight, bias, one_layer_weights=one_layer_weights,
                                          weight_scale_inv=weight_scale_inv)
                                _set_layernorm_func(layer_id, layernorm_weight, layernorm_bias,
                                                    one_layer_weights=one_layer_weights)
                        else:
                            if sub_key is not None:
                                if layer_name == ATTENTION_QKV_MAP and \
                                    ("is_layernorm" not in name_map[layer_name][sub_key] or \
                                     not name_map[layer_name][sub_key]["is_layernorm"]):
                                    _set_func(sub_key, layer_id, weight, bias, one_layer_weights=one_layer_weights,
                                              weight_scale_inv=weight_scale_inv)
                                else:
                                    _set_func(sub_key, layer_id, weight, bias, one_layer_weights=one_layer_weights)
                            else:
                                if layer_name == ATTENTION_DENSE:
                                    _set_func(layer_id, weight, bias,
                                              one_layer_weights=one_layer_weights, weight_scale_inv=weight_scale_inv)
                                else:
                                    _set_func(layer_id, weight, bias, one_layer_weights=one_layer_weights)
                                _set_layernorm_func(layer_id, layernorm_weight, layernorm_bias,
                                                    one_layer_weights=one_layer_weights)
                    if ep_id is not None and layer_name in [MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H]:
                        if first_k_dense_replace is not None and layer_id < first_k_dense_replace:
                            get_tensor_per_node(transformer, name_map[layer_name], is_moe_mlp=True,
                                                transpose_shape=tp_transpose_shape,
                                                weight_chunk_dim=weight_chunk_dim)
                        else:
                            for expert_id in ep_expert_mapping[ep_id]:
                                get_tensor_per_node(transformer, name_map[layer_name], expert_id=expert_id,
                                                    transpose_shape=etp_transpose_shape,
                                                    weight_chunk_dim=weight_chunk_dim)
                            if MOE_SHARED_EXPERT in name_map:
                                get_tensor_per_node(transformer, name_map[layer_name], is_shared=True,
                                                    transpose_shape=tp_transpose_shape,
                                                    weight_chunk_dim=weight_chunk_dim)
                    elif layer_name == MOE_GATE:
                        if first_k_dense_replace is None or layer_id >= first_k_dense_replace:
                            get_tensor_per_node(transformer, name_map[layer_name], transpose_shape=tp_transpose_shape,
                                                weight_chunk_dim=weight_chunk_dim)
                    else:
                        if sub_key is not None:
                            get_tensor_per_node(transformer, name_map[layer_name][sub_key]["name"],
                                                transpose_shape=tp_transpose_shape,
                                                weight_chunk_dim=weight_chunk_dim)
                        else:
                            get_tensor_per_node(transformer, name_map[layer_name], transpose_shape=tp_transpose_shape,
                                                weight_chunk_dim=weight_chunk_dim)

            if ep_id is None:
                if sub_key is None:
                    print(f"> p: {p}. {layer_name} chunk weight dim {weight_chunk_dim}, bias dim {bias_chunk_dim},"
                            f"max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                else:
                    print(f"> p: {p}. {layer_name}:{sub_key} chunk weight dim {weight_chunk_dim}, bias dim {bias_chunk_dim},"
                            f"max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
            else:
                if sub_key is None:
                    print(f"> p: {p}, ep_id: {ep_id}. {layer_name} chunk weight dim {weight_chunk_dim},"
                            f"bias dim {bias_chunk_dim},"
                            f"max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
                else:
                    print(f"> p: {p}, ep_id: {ep_id}. {layer_name}:{sub_key} chunk weight dim {weight_chunk_dim},"
                            f"bias dim {bias_chunk_dim},"
                            f"max_memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")

        def convert_one_p(p):
            if self.load_path is None and utils.LOADED_STATE_DICT is not None and \
                    p not in utils.LOADED_STATE_DICT.keys():
                return
            if utils.LOADED_LAYERS is not None:
                for stage_index in range(stage):
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, self)
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        utils.LOADED_LAYERS.add(layer_id)

            if cache_path is None:
                one_layer_weights_map = None
            else:
                one_layer_weights_map = {}
                is_done = True
                for stage_index in range(stage):
                    if not is_done:
                        break
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, self)
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        if not os.path.exists(f"{cache_path}/checkpoint_{layer_id}.pt") or \
                            not os.path.exists(f"{cache_path}/checkpoint_{layer_id}.done"):
                            is_done = False
                            if not is_done:
                                break
                if not is_done:
                    for stage_index in range(stage):
                        virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                            dualpipev, stage_index, p, pp, num_layers_in_rank, self)
                        for layer_index in range(num_layers_in_rank[virtual_p]):
                            layer_id = layer_index + layer_offset
                            if os.path.exists(f"{cache_path}/checkpoint_{layer_id}.done"):
                                os.remove(f"{cache_path}/checkpoint_{layer_id}.done")
                            if os.path.exists(f"{cache_path}/checkpoint_{layer_id}.pt"):
                                os.remove(f"{cache_path}/checkpoint_{layer_id}.pt")
                            one_layer_weights_map[layer_id] = {}
                if is_done and p != 0 and p != self.pp-1:
                    return

            state_dict_node = {}
            if etp is None:
                for t in range(self.tp):
                    key_e = 0
                    if self.load_path is None and utils.LOADED_STATE_DICT is not None and self.ep is not None:
                        key_e = list(utils.LOADED_STATE_DICT[p].keys())[0]
                    state_dict_node[t] = self.get_state_dict(p, t, e=None if self.ep is None else key_e)
            else:
                t_to_e_map = {}
                for ep_id in etp_to_tp_mapping:
                    for t in etp_to_tp_mapping[ep_id].values():
                        t_to_e_map[t] = ep_id
                for t in range(self.tp):
                    key_e = t_to_e_map[t]
                    state_dict_node[t] = self.get_state_dict(p, t, e=None if self.ep is None else key_e)

            if p == 0:
                # 1.1 word_embeddings
                if WORD_EMBEDDINGS in name_map:
                    parallel_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS}.weight")
                    name = self.get_word_embedding_name()
                    transformer = self.get_transformer_name(0)
                    weight, _ = self.get_tensor(state_dict_node, transformer, name + '.weight', parallel_dim)
                    if margs.get("add_embedding_padding", False) and weight is not None:
                        orig_vocab_size = cargs["vocab_size"]
                        weight = cut_embedding_padding(weight, orig_vocab_size)
                    c_ckpt.set_word_embedding(weight)
                    print(f"> p: {p}. word_embeddings weight: {weight.shape}") \
                        if weight is not None else None

                # 1.2 position embedding
                if margs.get("add_position_embedding", False):
                    name = self.get_position_embedding_name()
                    parallel_dim = tensor_parallel_dim.get(f"{WORD_POSITION_EMBEDDINGS}.weight")
                    weight, _ = self.get_tensor(state_dict_node, transformer, name + '.weight', parallel_dim)
                    c_ckpt.set_word_position_embedding(weight)
                    print(f"> p: {p}. add position embedding weight: {weight.shape}") \
                        if weight is not None else None

                if margs.get("add_block_position_embedding", False):
                    name =  self.get_block_position_embedding_name()
                    parallel_dim = tensor_parallel_dim.get(f"{WORD_BLOCK_POSITION_EMBEDDINGS}.weight")
                    weight, _ = self.get_tensor(state_dict_node, transformer, name + '.weight', parallel_dim)
                    c_ckpt.set_word_block_position_embedding(weight)
                    print(f"> p: {p}. add block position embedding weight: {weight.shape}") \
                        if weight is not None else None

                # 2. transformer layers

            # 2.1 input_layernorm
            if ep is None:
                if INPUT_LAYERNORM in name_map:
                    _convert(INPUT_LAYERNORM, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)
            else:
                _convert(INPUT_LAYERNORM, p, state_dict_node, ep_id=0, one_layer_weights_map=one_layer_weights_map)

            # 2.2 rotary_emb.inv_freqs
            if ATTENTION_ROTARY_EMB_INV_FREQ in name_map:
                chunk_dim = tensor_parallel_dim.get(ATTENTION_ROTARY_EMB_INV_FREQ)
                for stage_index in range(stage):
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, self)
                    if layer_for_test is not None:
                        cur_new_layer_index = 0
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        new_layer_index = layer_index
                        if layer_for_test is not None:
                            if layer_id in layer_for_test_map:
                                new_layer_index = cur_new_layer_index
                                cur_new_layer_index += 1
                            else:
                                continue
                        if min_mtp_layer_id is not None and layer_id >= min_mtp_layer_id:
                            cur_layer_prefix = mtp_layer_prefix
                        else:
                            cur_layer_prefix = layer_prefix

                        if one_layer_weights_map is None or layer_id in one_layer_weights_map:
                            one_layer_weights = one_layer_weights_map[layer_id] \
                                if one_layer_weights_map is not None else None

                            name = f"{cur_layer_prefix}.{new_layer_index}.{name_map[ATTENTION_ROTARY_EMB_INV_FREQ]}"
                            inv_freq, _ = self.get_tensor(state_dict_node, transformer, name, chunk_dim)
                            c_ckpt.set_layer_attention_rotary_emb_inv_freq(
                                layer_id, inv_freq, one_layer_weights=one_layer_weights)
                print(f"> p: {p}. rotary_emb.inv_freqs chunk dim {chunk_dim}")
            elif margs.get('use_rotary_position_embeddings', False):
                dim = hidden_size // num_attention_heads
                inv_freq = 1.0 / (margs.get('rotary_base', 10000) ** (torch.arange(0, dim, 2).float() / dim))
                for stage_index in range(stage):
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, self)
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        if one_layer_weights_map is None or layer_id in one_layer_weights_map:
                            one_layer_weights = one_layer_weights_map[layer_id] \
                                if one_layer_weights_map is not None else None
                            c_ckpt.set_layer_attention_rotary_emb_inv_freq(
                                layer_id, inv_freq, one_layer_weights=one_layer_weights)
                print(f"> p: {p}. rotary_emb.inv_freqs, created by dim {dim}")

            # 2.3 self attention query_key_value
            if ATTENTION_QUERY_KEY_VALUE in name_map:
                if ep is None:
                    _convert(ATTENTION_QUERY_KEY_VALUE, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)
                else:
                    _convert(ATTENTION_QUERY_KEY_VALUE, p, state_dict_node, ep_id=0,
                             one_layer_weights_map=one_layer_weights_map)
            if ATTENTION_QKV_MAP in name_map:
                for sub_key, dict_mcore in name_map[ATTENTION_QKV_MAP].items():
                    if ep is None:
                        _convert(ATTENTION_QKV_MAP, p, state_dict_node, sub_key=sub_key,
                                 one_layer_weights_map=one_layer_weights_map)
                    else:
                        _convert(ATTENTION_QKV_MAP, p, state_dict_node, ep_id=0, sub_key=sub_key,
                                 one_layer_weights_map=one_layer_weights_map)

            # 2.4 self attention dense
            if ep is None:
                _convert(ATTENTION_DENSE, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)
            else:
                _convert(ATTENTION_DENSE, p, state_dict_node, ep_id=0, one_layer_weights_map=one_layer_weights_map)

            # 2.5 post attention layernorm
            if ep is None:
                _convert(POST_ATTENTION_LAYERNORM, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)
            else:
                _convert(POST_ATTENTION_LAYERNORM, p, state_dict_node, ep_id=0, one_layer_weights_map=one_layer_weights_map)

            if ep is None:
                # 2.6 mlp dense h_to_4h
                if margs.get("transpose_mlp_dense", False):
                    _convert(MLP_DENSE_H_TO_4H, p, state_dict_node, tp_transpose_shape=(self.tp, 2),
                             one_layer_weights_map=one_layer_weights_map)
                else:
                    _convert(MLP_DENSE_H_TO_4H, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)
                # 2.7 mlp dense 4h_to_h
                _convert(MLP_DENSE_4H_TO_H, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)

                # 2.8 post mlp layernorm
                if POST_MLP_LAYERNORM in name_map:
                    _convert(POST_MLP_LAYERNORM, p, state_dict_node, one_layer_weights_map=one_layer_weights_map)
            elif one_layer_weights_map is None or len(one_layer_weights_map) > 0:
                _convert(MOE_GATE, p, state_dict_node, ep_id=0, one_layer_weights_map=one_layer_weights_map)
                for ep_id in range(ep):
                    if self.load_path is None and utils.LOADED_STATE_DICT is not None and \
                            ep_id not in utils.LOADED_STATE_DICT[p].keys():
                        continue
                    if self.etp is None:
                        expert_state_dict = {}
                        if ep_id == 0:
                            expert_state_dict = state_dict_node
                        else:
                            for t in range(self.tp):
                                expert_state_dict[t] = self.get_state_dict(p, t, e=ep_id)
                    else:
                        expert_state_dict = {}
                        for et in etp_to_tp_mapping[ep_id]:
                            t = etp_to_tp_mapping[ep_id][et]
                            expert_state_dict[et] = self.get_state_dict(p, t, e=ep_id)
                    # 2.6 mlp dense h_to_4h
                    if margs.get("transpose_mlp_dense", False):
                        etp_transpose_shape=(self.tp, 2)
                        if self.etp is not None:
                            etp_transpose_shape=(self.etp, 2)
                        _convert(MLP_DENSE_H_TO_4H, p, state_dict_node, tp_transpose_shape=(self.tp, 2),
                                 etp_transpose_shape=etp_transpose_shape, ep_id=ep_id,
                                 expert_state_dict=expert_state_dict, one_layer_weights_map=one_layer_weights_map)
                    else:
                        _convert(MLP_DENSE_H_TO_4H, p, state_dict_node, ep_id=ep_id,
                                 expert_state_dict=expert_state_dict, one_layer_weights_map=one_layer_weights_map)
                    # 2.7 mlp dense 4h_to_h
                    _convert(MLP_DENSE_4H_TO_H, p, state_dict_node, ep_id=ep_id, expert_state_dict=expert_state_dict,
                             one_layer_weights_map=one_layer_weights_map)

                    # 2.8 post mlp layernorm
                    if POST_MLP_LAYERNORM in name_map:
                        _convert(POST_MLP_LAYERNORM, p, state_dict_node, ep_id=0, one_layer_weights_map=one_layer_weights_map)
            add_final_layer = False
            if not dualpipev and p == pp - 1:
                add_final_layer = True
            elif dualpipev and p == 0:
                add_final_layer = True
            if add_final_layer:
                # 2.9 final_layernorm
                if FINAL_LAYERNORM in name_map:
                    name = name_map[FINAL_LAYERNORM]
                    transformer = self.get_transformer_name(self.num_stages - 1)

                    parallel_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.weight")
                    weight, _ =  self.get_tensor(state_dict_node, transformer, name + ".weight", parallel_dim)

                    parallel_dim = tensor_parallel_dim.get(f"{FINAL_LAYERNORM}.bias")
                    bias, _ =  self.get_tensor(state_dict_node, transformer, name + ".bias", parallel_dim)

                    c_ckpt.set_final_layernorm(weight, bias)
                    print(f"> p: {p}. final_layernorm weight {weight.shape}") if weight is not None else None

                # 3 word embedding for head
                if WORD_EMBEDDINGS_FOR_HEAD in name_map:
                    if margs.get("untie_embeddings_and_output_weights", False) or self.pp > 1:
                        parallel_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS_FOR_HEAD}.weight")
                        name = self.get_word_embedding_for_head_name()
                    else:
                        parallel_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS}.weight")
                        name = self.get_word_embedding_name()
                    weight, _ =  self.get_tensor(state_dict_node, transformer, name + '.weight', parallel_dim)
                    if margs.get("add_embedding_padding", False) and weight is not None:
                        orig_vocab_size = cargs["vocab_size"]
                        weight = cut_embedding_padding(weight, orig_vocab_size)
                    c_ckpt.set_word_embeddings_for_head(weight)
                    print(f"> p: {p}. word embedding for head weight {weight.shape}") if weight is not None else None

                if MTP_WORD_EMBEDDING in name_map:
                    virtual_p = pp * self.num_stages - 1
                    layer_offset = sum(num_layers_in_rank[:virtual_p])
                    transformer = self.get_transformer_name(self.num_stages - 1)
                    new_layer_index = 0
                    has_mtp = True
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        if layer_id < ori_num_layers:
                            continue
                        if layer_for_test is not None and layer_id not in layer_for_test_map:
                            continue

                        mtp_word_embedding, _ =  self.get_tensor(
                            state_dict_node, transformer, \
                            f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_WORD_EMBEDDING]}.weight", \
                            tensor_parallel_dim.get(f"{MTP_WORD_EMBEDDING}.weight"))
                        if mtp_word_embedding is None:
                            has_mtp = False
                            break
                        if margs.get("add_embedding_padding", False):
                            orig_vocab_size = cargs["vocab_size"]
                            mtp_word_embedding = cut_embedding_padding(mtp_word_embedding, orig_vocab_size)
                        mtp_enorm, _ =  self.get_tensor(
                            state_dict_node, transformer, \
                            f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_ENORM]}.weight", \
                            tensor_parallel_dim.get(f"{MTP_ENORM}.weight"))
                        mtp_hnorm, _ =  self.get_tensor(
                            state_dict_node, transformer, \
                            f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_HNORM]}.weight", \
                                tensor_parallel_dim.get(f"{MTP_HNORM}.weight"))
                        mtp_eh_proj, _ =  self.get_tensor(
                            state_dict_node, transformer, \
                            f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_EH_PROJ]}.weight", \
                                tensor_parallel_dim.get(f"{MTP_EH_PROJ}.weight"))
                        mtp_shared_head_norm, _ =  self.get_tensor(
                            state_dict_node, transformer, \
                            f"{mtp_layer_prefix}.{new_layer_index}.{name_map[MTP_SHARED_HEAD_NORM]}.weight", \
                                tensor_parallel_dim.get(f"{MTP_SHARED_HEAD_NORM}.weight"))

                        assert WORD_EMBEDDINGS_FOR_HEAD in name_map, \
                            f"{WORD_EMBEDDINGS_FOR_HEAD} is needed in name_map"
                        mtp_shared_head_head = c_ckpt.get_word_embeddings_for_head_weight()
                        new_layer_index += 1
                    if has_mtp and one_layer_weights_map is None or layer_id in one_layer_weights_map:
                        one_layer_weights = one_layer_weights_map[layer_id] \
                            if one_layer_weights_map is not None else None
                        c_ckpt.set_layer_mtp_weight(layer_id, mtp_word_embedding, mtp_enorm, mtp_hnorm, mtp_eh_proj,\
                                                    mtp_shared_head_norm, mtp_shared_head_head, \
                                                    one_layer_weights=one_layer_weights)
                        print(f"> p: {p}. mtp_word_embedding: {mtp_word_embedding.shape}, "\
                            f"mtp_enorm: {mtp_enorm.shape}, mtp_hnorm: {mtp_hnorm.shape}, "\
                            f"mtp_eh_proj: {mtp_eh_proj.shape}, mtp_shared_head_norm: {mtp_shared_head_norm.shape}, "\
                            f"mtp_shared_head_head: {mtp_shared_head_head.shape}")
            if cache_path is not None and one_layer_weights_map is not None and len(one_layer_weights_map) != 0:
                for stage_index in range(stage):
                    virtual_p, layer_offset, transformer = McoreCheckpoint.get_virtual_partition(
                        dualpipev, stage_index, p, pp, num_layers_in_rank, self)
                    for layer_index in range(num_layers_in_rank[virtual_p]):
                        layer_id = layer_index + layer_offset
                        torch.save(one_layer_weights_map[layer_id], f"{cache_path}/checkpoint_{layer_id}.pt")
                        done_file_name = f"{cache_path}/checkpoint_{layer_id}.done"
                        with open(done_file_name, 'w'):
                            os.utime(done_file_name, None)

        if args.max_workers > 1:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                for p in range(self.pp):
                    futures.append(executor.submit(convert_one_p, p=p))
            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    result = future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise e
        else:
            for p in range(self.pp):
                convert_one_p(p=p)

        # 4. optimizer
        if ep is not None:
            print("Optimizer is not supported in moe")
        elif self.has_optimizer():
            opt = merge_optimizer_by_pp_tp(self.optim_state_dict, margs)
            opt.build_param_map(self.get_named_parameters_shape())
            if margs.get("add_embedding_padding", False):
                orig_vocab_size = cargs["vocab_size"]
                opt.cut_embedding_padding(orig_vocab_size)
            if self.num_stages > 1:
                opt.interleave(self.pp, self.num_stages)
            if self.pp == 1 and not margs.get("untie_embeddings_and_output_weights", False):
                opt.add_word_embedding_for_head()
            c_ckpt.state_dict["optimizer"] = opt.to_dict()
            print("> optimizer params: ", opt.get_param_num())
        else:
            print("> optimizer empty")

        # 5.others
        c_ckpt.other_args["iteration"] = self.iteration
        c_ckpt.other_args["version"] = self.version
        c_ckpt.other_args["args"] = self.args
        c_ckpt.other_args["rng_state"] = self.rng_state

        return c_ckpt

    def set_name_map(self, name_map):
        """ set name_map """
        self.name_map = name_map

    def get_word_embedding_name(self):
        """ get word_embedding name """
        #if self.num_stages > 1:
        #    if WORD_EMBEDDINGS_TPL not in self.name_map:
        #        return None
        #    return self.name_map[WORD_EMBEDDINGS_TPL] % 0
        #else:
        return self.name_map.get(WORD_EMBEDDINGS)

    def get_position_embedding_name(self):
        """ get position_embedding name """
        #if self.num_stages > 1 :
        #    if WORD_POSITION_EMBEDDINGS_TPL not in self.name_map:
        #        return None
        #    return self.name_map[WORD_POSITION_EMBEDDINGS_TPL] % 0
        #else:
        return self.name_map.get(WORD_POSITION_EMBEDDINGS)

    def get_block_position_embedding_name(self):
        """ get block_position_embedding name """
        #if self.num_stages > 1 :
        #    if WORD_BLOCK_POSITION_EMBEDDINGS_TPL not in self.name_map:
        #        return None
        #    return self.name_map[WORD_BLOCK_POSITION_EMBEDDINGS_TPL] % 0
        #else:
        return self.name_map.get(WORD_BLOCK_POSITION_EMBEDDINGS)

    def get_transformer_name(self, stage_index):
        """ get transformer name """
        if self.num_stages > 1:
            return self.name_map[TRANSFORMER_TPL] % stage_index
        else:
            return self.name_map[TRANSFORMER]

    def get_word_embedding_for_head_name(self):
        """ get word_embedding for head name """
        #if self.num_stages > 1:
        #    return self.name_map[WORD_EMBEDDINGS_FOR_HEAD_TPL] % (self.num_stages - 1)
        #else:
        return self.name_map[WORD_EMBEDDINGS_FOR_HEAD]

    def has_optimizer(self):
        """ whether has optimizer """
        if self.optim_state_dict is None:
            return False
        for p in range(self.pp):
            for t in range(self.tp):
                if self.ep is None:
                    if self.optim_state_dict[p][t].empty():
                        return False
                else:
                    for ep_id in range(self.ep):
                        if self.optim_state_dict[p][ep_id][t].empty():
                            return False
        return True

    def has_bias(self, pp_rank=None):
        for p in range(self.pp) if pp_rank is None else [pp_rank]:
            for prefix, name, _ in self.layers[p]:
                if name.endswith("bias"):
                    return True
        return False

    def has_word_embeddings(self, p=None):
        """ whether has word embeddings """
        if WORD_EMBEDDINGS not in self.name_map:
            return False
        p = 0 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if prefix == self.get_word_embedding_name() and "weight" == name:
                return True
        return False
    def has_position_embeddings(self, p=None):
        """ whether has position embeddings """
        if WORD_POSITION_EMBEDDINGS not in self.name_map:
            return False
        p = 0 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if prefix == self.get_position_embedding_name() and "weight" == name:
                return True
        return False

    def has_block_position_embeddings(self, p=None):
        """ whether has block position embeddings """
        if WORD_BLOCK_POSITION_EMBEDDINGS not in self.name_map:
            return False
        p = 0 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if prefix == self.get_block_position_embedding_name() and "weight" == name:
                return True
        return False

    def has_word_embeddings_for_head(self, p=None):
        if WORD_EMBEDDINGS_FOR_HEAD not in self.name_map:
            return False
        p = self.pp-1 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if self.get_word_embedding_for_head_name() == prefix and "weight" == name:
                return True
        return False

    def has_final_layernorm(self, key, p=None):
        if FINAL_LAYERNORM not in self.name_map:
            return False
        p = self.pp-1 if p is None else p
        for prefix, name, _ in self.layers[p]:
            if f"{self.name_map[FINAL_LAYERNORM]}.{key}" == name:
                return True
        return False

    def get_layers(self, pp_rank=None):
        """ get all layers """
        layers = []
        tensor_parallel_dim = self.tensor_parallel_dim

        # embedding
        if pp_rank == 0 or pp_rank is None:
            keys = (WORD_EMBEDDINGS, \
                    WORD_POSITION_EMBEDDINGS, \
                    WORD_BLOCK_POSITION_EMBEDDINGS)
            names = [self.get_word_embedding_name(), \
                    self.get_position_embedding_name(), \
                    self.get_block_position_embedding_name()]
            if self.ep is None:
                state_dict = self.get_state_dict(0, 0)
            else:
                state_dict = self.get_state_dict(0, 0, 0)
            for i in range(3):
                if names[i] is not None and check_path_in_dict(state_dict, names[i]):
                    chunk_dim = tensor_parallel_dim.get(f"{keys[i]}.weight", -1)
                    layers.append((names[i], "weight", chunk_dim))

        # transformers
        if self.ep is None:
            if ATTENTION_QUERY_KEY_VALUE in self.name_map:
                TRANSFORMER_LAYERS = [INPUT_LAYERNORM, ATTENTION_QUERY_KEY_VALUE, ATTENTION_DENSE, \
                        POST_ATTENTION_LAYERNORM, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, ]
            else:
                TRANSFORMER_LAYERS = [INPUT_LAYERNORM, ATTENTION_QKV_MAP, ATTENTION_DENSE, \
                        POST_ATTENTION_LAYERNORM, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, ]
        else:
            if ATTENTION_QUERY_KEY_VALUE in self.name_map:
                TRANSFORMER_LAYERS = [INPUT_LAYERNORM, ATTENTION_QUERY_KEY_VALUE, ATTENTION_DENSE, \
                        POST_ATTENTION_LAYERNORM, MOE_GATE, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, ]
            else:
                TRANSFORMER_LAYERS = [INPUT_LAYERNORM, ATTENTION_QKV_MAP, ATTENTION_DENSE, \
                        POST_ATTENTION_LAYERNORM, MOE_GATE, MLP_DENSE_H_TO_4H, MLP_DENSE_4H_TO_H, ]

        num_layers_per_pp = self.num_layers // self.pp
        num_layers_per_stage = num_layers_per_pp // self.num_stages
        for p in range(self.pp) if pp_rank is None else [pp_rank]:
            layer_prefix = self.name_map.get(LAYER_PREFIX)
            if self.ep is None:
                state_dict = self.get_state_dict(p, 0)
            else:
                state_dict = self.get_state_dict(p, 0, 0)
            for layer_id in range(num_layers_per_pp):
                layer_prefix = self.name_map.get(LAYER_PREFIX)
                stage_index = layer_id // num_layers_per_stage
                layer_index = layer_id % num_layers_per_stage
                transformer_name = self.get_transformer_name(stage_index)
                transformer = get_element_from_dict_by_path(state_dict, transformer_name)
                for layer in TRANSFORMER_LAYERS:
                    for w in ("weight", "bias"):
                        layer_name = f"{layer_prefix}.{layer_index}.{self.name_map[layer]}.{w}"
                        if layer_name in transformer.keys():
                            chunk_dim = tensor_parallel_dim.get(f"{layer}.{w}", -1)
                            layers.append((transformer_name, layer_name, chunk_dim))
                for layer in [ATTENTION_ROTARY_EMB_INV_FREQ]:
                    if layer not in self.name_map:
                        continue
                    layer_name = f"{layer_prefix}.{layer_index}.{self.name_map[layer]}"
                    if layer_name in transformer.keys():
                        chunk_dim = self.tensor_parallel_dim.get(f"{layer}", -1)
                        layers.append((transformer_name, layer_name, chunk_dim))

            transformer_name = self.get_transformer_name(self.num_stages - 1)
            transformer = get_element_from_dict_by_path(state_dict, transformer_name)
            for layer in [FINAL_LAYERNORM] if FINAL_LAYERNORM in self.name_map else []:
                for w in ("weight", "bias"):
                    layer_name = f"{self.name_map[layer]}.{w}"
                    if layer_name in transformer.keys():
                        chunk_dim = tensor_parallel_dim.get(f"{layer}.{w}", -1)
                        layers.append((transformer_name, layer_name, chunk_dim))

        # emebdding for head
        if WORD_EMBEDDINGS_FOR_HEAD in self.name_map:
            if pp_rank == self.pp-1 or pp_rank is None:
                layer = self.get_word_embedding_for_head_name()
                chunk_dim = tensor_parallel_dim.get(f"{WORD_EMBEDDINGS_FOR_HEAD}.weight", -1)
                layers.append((layer, "weight", chunk_dim))

        return layers

    def get_transformer_layers(self, pp_rank, layer_index):
        """ get transformer layers """
        layers = []
        pp_rank = 0 if pp_rank is None else pp_rank
        transformer_name = self.get_transformer_name(0)
        for meta in self.get_named_parameters_shape(pp_rank):
            layer_name = meta[0]
            key = f"{transformer_name}.{self.name_map[LAYER_PREFIX]}.{layer_index}."
            if key in layer_name:
                layers.append(meta)
        # print(pp_rank, layer_index, layers)
        return layers

    def get_named_parameters_shape(self, p=None, t=None):
        if p == None and t == None:
            return self.named_parameters_shape
        if p != None and t != None:
            return self.named_parameters_shape_by_pt[p][t]
        if p != None:
            return self.named_parameters_shape_by_p[p]
        if t != None:
            return self.named_parameters_shape_by_t[t]

    def _get_named_parameters_shape(self, pp_rank=None, tp_rank=None):
        """ return list of (layer_name, tensor_shape, parallel_dim) """
        result = []
        for p in range(self.pp) if pp_rank is None else [pp_rank]:
            if self.ep is None:
                state_dict = self.get_state_dict(p, 0)
            else:
                state_dict = self.get_state_dict(p, 0, 0)
            for layer, key, parallel_dim in self.layers[p]:
                if key.endswith("weight") or key.endswith("bias"):
                    element = get_element_from_dict_by_path(state_dict, layer)
                    if key in element:
                        assert element[key] is not None, f"key {key}"
                        shape = element[key].shape
                        if tp_rank is None and parallel_dim >= 0:
                            shape = list(shape)
                            shape[parallel_dim] *= self.tp
                            shape = torch.Size(shape)
                        result.append((f"{layer}.{key}", shape, parallel_dim))
        return result

    def load(self, load_path, m_config, name_map, load_optimizer=True):
        """
        Load mcore checkpoint from checkpoints folder.

            Args:
                load_path (str): the path to checkpoint
                m_config: mcore m_config loaded from ckpt
        """

        args = parse_args()
        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        dp = args.data_parallel_size
        ep = args.expert_parallel_size
        etp = args.expert_tensor_parallel_size if hasattr(args, 'expert_tensor_parallel_size') else None
        dtype = m_config.get("dtype")
        tensor_parallel_dim = m_config.get("tensor_parallel_dim")
        num_layers_per_stage = m_config.get('num_layers_per_virtual_pipeline_stage')
        custom_pipeline_layers = m_config.get("custom_pipeline_layers")
        stage = args.num_virtual_stages_per_pipeline_rank
        if stage is None:
            stage = None if num_layers_per_stage is None else (self.num_layers // pp // num_layers_per_stage)
        use_distributed_optimizer = m_config.get("use_distributed_optimizer", False)
        self.set_dtype(dtype)
        self.init_pipeline_size(pp, tp, dp, ep, tensor_parallel_dim, stage,
                                custom_pipeline_layers=custom_pipeline_layers, etp=etp)
        self.set_name_map(name_map)

        key_t = 0
        key_p = 0
        key_e = 0
        if self.load_path is None and utils.LOADED_STATE_DICT is not None:
            key_p = list(utils.LOADED_STATE_DICT.keys())[0]
            if self.ep is not None:
                key_e = list(utils.LOADED_STATE_DICT[key_p].keys())[0]

        if self.ep is None:
            state_dict = self.get_state_dict(key_p, key_t)
        else:
            state_dict = self.get_state_dict(key_p, key_t, key_e)
        self.iteration = state_dict.get('iteration', 0)
        self.version = state_dict['checkpoint_version']
        self.args = state_dict['args']
        self.rng_state = state_dict.get('rng_state', None)

        self.optim_state_dict = None
        # optimizer
        if load_optimizer:
            self.init_layers()
            self.init_named_parameters_shape()
            self.init_optimizer(use_distributed_optimizer)
            if use_distributed_optimizer:
                for p in range(self.pp):
                    for t in range(self.tp):
                        opts = []
                        named_parameters_shape = self.get_named_parameters_shape(p, t)
                        if self.ep is None:
                            for d in range(self.dp):
                                if self.pp == 1:
                                    checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}"
                                else:
                                    checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}"
                                checkpoint_dir = os.path.join(load_path, checkpoint_dir)
                                checkpoint_path = os.path.join(checkpoint_dir, "distrib_optim.pt")
                                optim_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                                opt = MegatronOptimizer.generate_optimizer(self, self.num_layers // self.pp, p)
                                opt.load(optim_state_dict)
                                opts.append(opt)
                                opt.debug(f"tp/pp/dp rank: {t}/{p}/{d}, load from: {checkpoint_path}")
                            self.optim_state_dict[p][t] = merge_optimizer_by_dp(opts, named_parameters_shape)
                            self.optim_state_dict[p][t].debug(f"merge by dp {self.dp} in pp/tp rank {p}/{t}")
                        else:
                            for e in range(self.ep):
                                for d in range(self.dp):
                                    if self.pp == 1:
                                        checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}_{e:03d}"
                                    else:
                                        checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}_{e:03d}"
                                    checkpoint_dir = os.path.join(load_path, checkpoint_dir)
                                    checkpoint_path = os.path.join(checkpoint_dir, "distrib_optim.pt")
                                    optim_state_dict = torch.load(
                                        checkpoint_path, map_location="cpu", weights_only=False)
                                    opt = MegatronOptimizer.generate_optimizer(self, self.num_layers // self.pp, p)
                                    opt.load(optim_state_dict)
                                    opts.append(opt)
                                    opt.debug(f"tp/pp/ep/dp rank: {t}/{p}/{e}{d}, load from: {checkpoint_path}")
                                self.optim_state_dict[p][e][t] = merge_optimizer_by_dp(opts, named_parameters_shape)
                                self.optim_state_dict[p][e][t].debug(f"merge by dp {self.dp} in pp/tp/ep rank {p}/{t}/{e}")

            else:
                for p in range(self.pp):
                    for t in range(self.tp):
                        if self.ep is None:
                            state_dict = self.get_state_dict(p, t)
                            if "optimizer" in state_dict:
                                named_parameters_shape = self.get_named_parameters_shape(p, t)
                                self.optim_state_dict[p][t].load(state_dict, named_parameters_shape)
                        else:
                            for e in range(self.ep):
                                state_dict = self.get_state_dict(p, t, e)
                                if "optimizer" in state_dict:
                                    named_parameters_shape = self.get_named_parameters_shape(p, t)
                                    self.optim_state_dict[p][e][t].load(state_dict, named_parameters_shape)
        self.debug("==================== mcore checkpoint loaded ================================")

    def pre_save(self, save_path, m_config=None):
        """
        Before saving the model, delete the old save directory,
        create a new save directory, and update the tracking file.
        If 'm_config' is not provided, the current 'mcore' configuration will be used.

        Args:
            save_path (str): Path where the model should be saved.
            m_config (Optional[dict], optional): Optional `mcore` configuration dictionary, default to None.

        Returns:
            tuple(str, dict): Returns a tuple containing two elements: the first is the new saved directory path,
                and the second is the updated `mcore` configuration dictionary.
        """
        os.makedirs(save_path, exist_ok=True)
        # Saving the tracker file
        tracker_filepath = os.path.join(save_path, "latest_checkpointed_iteration.txt")
        with open(tracker_filepath, "w") as f:
            f.write(str(self.iteration or "release"))

        # create `release` dir in args.load_path
        folder_name = f"iter_{self.iteration:07d}" if self.iteration > 0 else "release"
        release_dir = os.path.join(save_path, folder_name)
        os.makedirs(release_dir, exist_ok=True)

        # mcore config
        margs = self.args
        if m_config is not None:
            for k, v in m_config.data.items():
                setattr(margs, k, v)
        print(f"Saving mcore args {margs}")
        return release_dir, margs

    def save_model_file(self, release_dir, margs, p, t, e, state_dict_node, optim_state_dict_node, saved_models_str):
        """
        Save the model file, including model parameters, optimizer state, and random seed.
        If the number of iterations is None, use mp_rank as the directory name; otherwise,
        use mp_rank and epoch as the directory name.

        Args:
            release_dir (str): The path of the release directory.
            margs (Optional[Namespace], optional): Namespace object of command line parameters, default is None.
            p (int): process number mp_rank.
            t (int): task number mp_rank.
            e (Optional[int], optional): The number of epochs, default to None.
            state_dict_node (Dict[str, Any]): Model parameter dictionary.
            optim_state_dict_node (Dict[str, Any]): Optimizer state dictionary.

        Returns:
            None.

        Raises:
            None.
        """
        state_dict_node["checkpoint_version"] = self.version
        if e is None or self.ep == 1:
            checkpoint_dir = (
                f"mp_rank_{t:02d}"
                if self.pp == 1
                else f"mp_rank_{t:02d}_{p:03d}"
            )
        else:
            checkpoint_dir = (
                f"mp_rank_{t:02d}_{e:03d}"
                if self.pp == 1
                else f"mp_rank_{t:02d}_{p:03d}_{e:03d}"
            )

        checkpoint_name = "model_optim_rng.pt"
        if optim_state_dict_node is not None:
            state_dict_node.update(optim_state_dict_node.to_dict())
        if margs is not None:
            state_dict_node['args'] = margs
        if self.rng_state is not None:
            state_dict_node['rng_state'] = self.rng_state
        state_dict_node["iteration"] = self.iteration
        checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(state_dict_node, checkpoint_path)
        print(f"Saving mcore checkpoint {state_dict_node.keys()} to: {checkpoint_path}, {saved_models_str}")

    def save_optimizer(self, release_dir, p, t, e, optim_state_dict_node):
        """
        Save the optimizer state.
        If the current epoch is None, the name is in the mp_rank_t_d format; otherwise,
        the name is in the mp_rank_t_e_d format.
        Each distributed process saves its optimizer state to a different directory.

        Args:
            release_dir (str) The directory path where the optimizer state should be saved.
            p (int) The current process number starts from 0.
            t (int) The current task number starts from 0.
            e (Optional[int]) The current epoch is optional, and the default is None.
            optim_state_dict_node (DistributedOptimizerStateDictNode)
                DistributedOptimizerStateDictNode object containing the optimizer state.

        Returns:
            None.
        """
        chunk_optimers = optim_state_dict_node.chunk_by_dp(self.dp, self.num_stages)
        for d in range(self.dp):
            if e is None or self.ep == 1:
                if self.pp == 1:
                    checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}"
                else:
                    checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}"
            else:
                if self.pp == 1:
                    checkpoint_dir = f"mp_rank_{t:02d}_{d:03d}_{e:03d}"
                else:
                    checkpoint_dir = f"mp_rank_{t:02d}_{p:03d}_{d:03d}_{e:03d}"

            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "distrib_optim.pt")
            torch.save(
                chunk_optimers[d].to_dict(),
                checkpoint_path,
            )
            if e is None or self.ep == 1:
                chunk_optimers[d].debug(f"tp/pp/dp rank {t}/{p}/{d}, saved to: {checkpoint_path}")
            else:
                chunk_optimers[d].debug(f"tp/pp/ep/dp rank {t}/{p}/{e}/{d}, saved to: {checkpoint_path}")

    def debug(self, title):
        """ debbug """
        print(f"\n【Mcore】{title}")
        if self.ep is None:
            print(f"-> tp/pp/dp size: {self.tp}/{self.pp}/{self.dp}")
        else:
            print(f"-> tp/pp/ep/dp size: {self.tp}/{self.pp}/{self.ep}/{self.dp}")
        if self.has_optimizer():
            for t in range(self.tp):
                for p in range(self.pp):
                    if self.ep is None:
                        self.optim_state_dict[p][t].debug(f"tp/pp rank {t}/{p}")
                    else:
                        for e in range(self.ep):
                            self.optim_state_dict[p][e][t].debug(f"tp/pp/ep rank {t}/{p}/{e}")

        print("\n")

if __name__ == "__main__":
    pass
