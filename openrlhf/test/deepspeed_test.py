import io
import logging
import time
from datetime import timedelta

from click import group
import deepspeed
from deepspeed.runtime import zero
from torch.optim import Optimizer
import ray
import ray.util.collective as collective
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from typing import Any, Optional, Union
import os
import socket
from typing import Callable, Dict, List, Optional, Type
from deepspeed.monitor.monitor import MonitorMaster
import torch.distributed as dist
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
#from openrlhf.utils import DeepspeedStrategy, blending_datasets, get_tokenizer
#from openrlhf.datasets import PromptDataset, SFTDataset
from deepspeed.utils import groups
from deepspeed.accelerator import get_accelerator
from deepspeed.constants import TORCH_DISTRIBUTED_DEFAULT_PORT
from deepspeed.runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
from deepspeed.runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.inference.engine import InferenceEngine
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.runtime.lr_schedules import add_tuning_arguments
from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from deepspeed.utils import log_dist, OnDevice, logger
from deepspeed.module_inject import replace_transformer_layer, revert_transformer_layer
from torch.optim.lr_scheduler import _LRScheduler
from deepspeed.utils.debug import debug_extract_module_and_param_names, debug_clear_module_and_param_names
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _store_based_barrier,
    _world,
    default_pg_timeout,
    rendezvous,
    gather_object,
    GroupMember,
    get_rank
)
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.utils.timer import NoopTimer, ThroughputTimer, SynchronizedWallClockTimer, \
    FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER, \
    STEP_MICRO_TIMER, \
    FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_GLOBAL_TIMER

from deepspeed.runtime.data_pipeline.constants import DATA_SAMPLING, \
    DATA_ROUTING, DATA_SAMPLING_ENABLED, CURRICULUM_LEARNING, \
    CURRICULUM_LEARNING_ENABLED, DATA_SAMPLING_NUM_WORKERS, RANDOM_LTD, \
    RANDOM_LTD_ENABLED, RANDOM_LTD_LAYER_ID, RANDOM_LTD_LAYER_NUM, \
    RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE, RANDOM_LTD_LAYER_TOKEN_LR_ENABLED, \
    RANDOM_LTD_GLOBAL_BATCH_SIZE, RANDOM_LTD_MICRO_BATCH_SIZE, DATA_EFFICIENCY

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.runtime.engine import EngineTimers

from torch._C._distributed_c10d import (
    _resolve_process_group)

# rewrite deepspeed initialize function

def initialize(args=None,
               distributed_port: int = TORCH_DISTRIBUTED_DEFAULT_PORT,
               mpu=None,
               dist_init_required: Optional[bool] = None,
               config=None,
               mesh_param=None,
               config_params=None):
   
    #log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(__version__, __git_hash__, __git_branch__), ranks=[0])

    # Disable zero.Init context if it's currently enabled
    start = time.time()
    zero.partition_parameters.shutdown_init_context()
    last = time.time() - start
    
    print(" Zero.partition_parameters.shutdown_init_context Last", last)

    global dist
    from deepspeed import comm as dist
    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend,
                          distributed_port=distributed_port,
                          dist_init_required=dist_init_required)

    ##TODO: combine reuse mpu as mesh device and vice versa
    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    mesh_device = None
    if mesh_param:
        logger.info(f"mesh_param to Initialize mesh device: {mesh_param}")
        mesh_device = dist.initialize_mesh_device(mesh_param, ("data_parallel", "sequence_parallel"))
    #if config file has sequence parallelize and data parallelize, then use them to initialize mesh device
    elif config is not None:
        if "sequence_parallel_size" in config and "data_parallel_size" in config:
            logger.info(f"config to Initialize mesh device: {config}")
            mesh_device = dist.initialize_mesh_device((config["data_parallel_size"], config["sequence_parallel_size"]), \
            ("data_parallel", "sequence_parallel"))

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        logger.warning("************ --deepscale_config is deprecated, please use --deepspeed_config ************")
        if hasattr(args, "deepspeed_config"):
            assert (args.deepspeed_config
                    is None), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert config is None, "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    return args, config, mpu, mesh_device, 


def init_deepspeed_second_step(self, model):
    args, config, mpu, mesh_device = initialize(model=model, config=ds_config)
    config_class = DeepSpeedConfig(config, mpu, mesh_device=mesh_device)
        # TODO rewrite the deepspeedengine
    engine = DynamicEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=mpu,
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                mesh_device=mesh_device,
                                config_class=config_class)
    
    

    # Restore zero.Init context if necessary
    zero.partition_parameters.restore_init_context()

    return_items = [
            engine,
            engine.optimizer,
            engine.training_dataloader,
            engine.lr_scheduler,
        ]
    return tuple(return_items)    



# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)
    print("Store", store)

    # Ensure global _world.pg_group_ranks is initialized
    from torch.distributed.distributed_c10d import _world
    if not hasattr(_world, "pg_group_ranks"):
        _world.pg_group_ranks = {}
    
    
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    print("---PG---", pg.name())
    
    if pg is not None:
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        print("New rank is", get_rank(pg))
        print(f"[init_process_group] Process group registered: {pg}, rank: {rank}")
        
    # if GroupMember.WORLD is None:
    #     GroupMember.WORLD = pg

    return pg




class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)  
        self.labels = torch.randint(0, 2, (size,))  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True  
    },
    "zero_optimization": {
        "stage": 3
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-3
        }
    }
}


class DynamicEngine(DeepSpeedEngine):
    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,
                 config_class=None,
                 mesh_device=None,
                 dont_change_device=False,
                 using_existing_engine = False):
        super(DeepSpeedEngine, self).__init__()
        self.dont_change_device = dont_change_device
        self.client_optimizer = optimizer
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.all_to_all_group = None
        self.data_parallel_group = None
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config = config
        self._config = config_class
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None
        self.enable_backward_allreduce = True
        self.progressive_layer_drop = None
        self.eigenvalue = None
        self.block_eigenvalue = None
        self.gas_boundary_ctr = 0
        self.dist_backend = get_accelerator().communication_backend_name()
        self.has_moe_layers = False
        self.num_experts = []
        self.gate_modules = []
        self.moe_layers = []
        self._step_applied = False
        self._global_grad_norm = None
        self.use_ds_comm = False  # False --> Use torch.dist, True --> Use ds.comm backend.

        self.checkpoint_engine = None

        self._is_gradient_accumulation_boundary = None
        self.scale_wrt_gas = None
        self.losses = None
        self.mesh_device = mesh_device

        # for debug purposes - can then debug print: debug_get_module_name(module)        
        
        debug_extract_module_and_param_names(model)

        if self.mesh_device:
            groups.mesh_device = self.mesh_device

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()
        self._set_distributed_vars(args)

        dist.configure(self._config)

        self.monitor = MonitorMaster(self._config.monitor_config)


        self.pipeline_parallelism = False

        # Using existing engine or not
        if using_existing_engine:
            self.using_existing_model()
        else:
            self._configure_distributed_model(model)

            # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
            self.param_names = {param: name for name, param in model.named_parameters()}

            self._get_model_parameters()

            # Configure wall clock timers
            self.timers = SynchronizedWallClockTimer()
            # Throughput timer
            self.tput_timer = ThroughputTimer(self._config.timers_config,
                                            batch_size=self.train_batch_size(),
                                            steps_per_output=self.steps_per_print(),
                                            monitor_memory=False)

            log_dist(f"DeepSpeed Flops Profiler Enabled: {self.flops_profiler_enabled()}", ranks=[0])

            if self.flops_profiler_enabled():
                self.flops_profiler = FlopsProfiler(self.module, self, self.flops_profiler_recompute_fwd_factor())

            if training_data:
                self.training_dataloader = self.deepspeed_io(training_data)
            else:
                self.training_dataloader = None

            # Configure optimizer and scheduler
            self.optimizer = None
            self.basic_optimizer = None
            self.lr_scheduler = None
            has_optimizer = False

            if optimizer or self.optimizer_name():
                has_optimizer = True
            # If no parameters given by init default to module parameters
            if model_parameters is None:
                model_parameters = self.module.parameters()

            # Convert model parameters from generator to list
            if not isinstance(model_parameters, list):
                model_parameters = list(model_parameters)

            if has_optimizer:
                self._configure_optimizer(optimizer, model_parameters)
                self._configure_lr_scheduler()
                self._report_progress(0)
            elif self.zero_optimization():
                # no optim selected but zero is enabled
                self.optimizer = self._configure_zero_optimizer(optimizer=None) #Optimizer
            elif self.bfloat16_enabled():
                self.optimizer = self._configure_bf16_optimizer(optimizer=None)

            # Hook optimizer for snip_momentum pruning
            if hasattr(model, 'pruners'):
                from deepspeed.compression.helper import rewrite_optimizer_step
                self.optimizer.pruners = model.pruners
                rewrite_optimizer_step(self.optimizer)

            # Bookkeeping for sparse support
            self.sparse_tensor_module_names = set()
            # if self.sparse_gradients_enabled():
            for name, module in self.module.named_modules():
                if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                    self.sparse_tensor_module_names.add(name + ".weight")
                    logger.info("Will convert {} to sparse tensor during training".format(name))

            self._optimized_linear_offload_setup()

            self.save_non_zero_checkpoint = False
            self.save_zero_checkpoint = False
            if not isinstance(self.optimizer, DeepSpeedZeRoOffload):
                self._configure_checkpointing(dist_init_required)

            if self.eigenvalue_enabled():
                self.eigenvalue = self._configure_eigenvalue()

            if self.pld_enabled():
                self.progressive_layer_drop = self._configure_progressive_layer_drop()

            if self.curriculum_enabled_legacy():
                self.curriculum_scheduler_legacy = self._configure_curriculum_scheduler_legacy()

            if self.random_ltd_enabled():
                random_ltd_config = self.random_ltd_config()
                random_ltd_config[RANDOM_LTD_GLOBAL_BATCH_SIZE] = self.train_batch_size()
                random_ltd_config[RANDOM_LTD_MICRO_BATCH_SIZE] = self.train_micro_batch_size_per_gpu()
                self.random_ltd_scheduler = self._configure_random_ltd_scheduler(random_ltd_config)

            # Engine timers
            self.engine_timers = EngineTimers(enable_micro_timers=self.wall_clock_breakdown(),
                                            enable_global_timers=self.wall_clock_breakdown()
                                            or self.flops_profiler_enabled())

            # Use torch (un)flatten ops
            self.flatten = _flatten_dense_tensors
            self.unflatten = _unflatten_dense_tensors

            self._is_compiled = False

    
    def using_existing_model(self, model, optimizer, training_data):
        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self._configure_existing_distributed_model(model)
        
        self.param_names = {param: name for name, param in model.named_parameters()}

        self._get_model_parameters()

        # Configure wall clock timers
        # self.timers = SynchronizedWallClockTimer()
        # # Throughput timer
        # self.tput_timer = ThroughputTimer(self._config.timers_config,
        #                                   batch_size=self.train_batch_size(),
        #                                   steps_per_output=self.steps_per_print(),
        #                                   monitor_memory=False)

        log_dist(f"DeepSpeed Flops Profiler Enabled: {self.flops_profiler_enabled()}", ranks=[0])

        if self.flops_profiler_enabled():
            self.flops_profiler = FlopsProfiler(self.module, self, self.flops_profiler_recompute_fwd_factor())

        if training_data:
            self.training_dataloader = self.deepspeed_io(training_data)
        else:
            self.training_dataloader = None

        # Configure optimizer and scheduler
        self.optimizer = None
        self.basic_optimizer = None
        self.lr_scheduler = None
        has_optimizer = False

        if optimizer or self.optimizer_name():
            has_optimizer = True
        # If no parameters given by init default to module parameters
        if model_parameters is None:
            model_parameters = self.module.parameters()

        # Convert model parameters from generator to list
        if not isinstance(model_parameters, list):
            model_parameters = list(model_parameters)

        if has_optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler()
            self._report_progress(0)
        elif self.zero_optimization():
            # no optim selected but zero is enabled
            self.optimizer = self._configure_zero_optimizer(optimizer=None) #Optimizer
        elif self.bfloat16_enabled():
            self.optimizer = self._configure_bf16_optimizer(optimizer=None)

        # Hook optimizer for snip_momentum pruning
        
        # if hasattr(model, 'pruners'):
        #     from ..compression.helper import rewrite_optimizer_step
        #     self.optimizer.pruners = model.pruners
        #     rewrite_optimizer_step(self.optimizer)

        # Bookkeeping for sparse support
        self.sparse_tensor_module_names = set()
        # if self.sparse_gradients_enabled():
        for name, module in self.module.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                self.sparse_tensor_module_names.add(name + ".weight")
                logger.info("Will convert {} to sparse tensor during training".format(name))

        self._optimized_linear_offload_setup()

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        if not isinstance(self.optimizer, DeepSpeedZeRoOffload):
            self._configure_checkpointing(dist_init_required)

        if self.eigenvalue_enabled():
            self.eigenvalue = self._configure_eigenvalue()

        if self.pld_enabled():
            self.progressive_layer_drop = self._configure_progressive_layer_drop()

        if self.curriculum_enabled_legacy():
            self.curriculum_scheduler_legacy = self._configure_curriculum_scheduler_legacy()

        if self.random_ltd_enabled():
            random_ltd_config = self.random_ltd_config()
            random_ltd_config[RANDOM_LTD_GLOBAL_BATCH_SIZE] = self.train_batch_size()
            random_ltd_config[RANDOM_LTD_MICRO_BATCH_SIZE] = self.train_micro_batch_size_per_gpu()
            self.random_ltd_scheduler = self._configure_random_ltd_scheduler(random_ltd_config)

        # Engine timers

        # self.engine_timers = EngineTimers(enable_micro_timers=self.wall_clock_breakdown(),
        #                                   enable_global_timers=self.wall_clock_breakdown()
        #                                   or self.flops_profiler_enabled())

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        self._is_compiled = False


    def _configure_existing_distributed_model(self, model):
        self._set_client_model(model)
        is_zero_init_model = self.zero_optimization_partition_weights() and any(
            [hasattr(param, "ds_id") for param in self.module.parameters()])

        # zero.Init() handles device placement of model

        if not (self.dont_change_device or is_zero_init_model):
            self.module.to(self.device)

        # Set deepspeed parallelism spec. for the model including expert parallelism
        for _, module in self.module.named_modules():
            if hasattr(module, 'set_deepspeed_parallelism'):
                module.set_deepspeed_parallelism(self._config.use_data_before_expert_parallel_)

        # Query the groups module to get information about various parallel groups
        self.local_all_to_all_group = None
        if self.zero_quantized_gradients():
            log_dist("Using quantized gradients", ranks=[0])
            self.local_all_to_all_group = groups._get_local_all_to_all_group()
        self.data_parallel_group = groups._get_data_parallel_group()
        self.dp_world_size = groups._get_data_parallel_world_size()
        self.seq_data_parallel_group = groups._get_sequence_data_parallel_group()
        self.seq_dp_world_size = groups._get_sequence_data_parallel_world_size()
        self.mp_world_size = groups._get_model_parallel_world_size()
        self.expert_parallel_group = groups._get_expert_parallel_group_dict()
        self.expert_data_parallel_group = groups._get_expert_data_parallel_group_dict()
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        if self.sequence_parallel_size > 1:
            self.communication_data_type = self._config.seq_parallel_communication_data_type
            self.seq_parallel_group = groups._get_sequence_parallel_group()
        
        if not (self.amp_enabled() or is_zero_init_model):
            self._broadcast_model()



@ray.remote(num_gpus=1)
class DistributedRayActor:
    def __init__(self, rank, world_size, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = str(self._rank % torch.cuda.device_count())
        
        

    # def compute_sum(self, tensor):
    #     tensor = tensor.to(torch.device(f"cuda:{torch.cuda.current_device()}")) 
    #     dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
    #     return tensor.cpu()  
    
    
    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_current_addr_port(self):
        addr = self._get_current_node_ip()
        port = self._get_free_port()
        
        return addr, port

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    # must be called after init deepspeed
    def get_engine_optimizer(self):
        return self.model_engine, self.optimizer


    def get_rank(self):
        return get_rank()
    
    def get_group_rank(self):
        return get_rank(self.new_group)
    
    
    def init_ray_collective_group(self, backend, world_size, rank, group_name):
        self.new_group = collective.init_collective_group(backend=backend, world_size=world_size, rank=rank, group_name=group_name)
    
    
    def init_deepspeed(self, actor=None, model = None):
        self.model = model
        
        start = time.time()
        
        if actor == None:
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
            last = time.time() - start
            print("----------Deepspeed Initialized Last: ", last)
            
        print(self.model_engine)
    
    def init_deepspeed_first_step(self):
        initialize(config=ds_config)
    
    
    def init_process_group(self, 
        backend,
        init_method,
        world_size,
        rank,
        group_name):
        print("Start Init Process Group")
        self.new_group = init_process_group(backend=backend,init_method=init_method, world_size=world_size,rank=rank,group_name=group_name)
        print("New Group", self.new_group)
        print("New Group Rank", get_rank(self.new_group))
        
        
    def init_torch_group(self, rank, world_size, master_addr, master_port):
        self.global_group = dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
        
    # def create_new_instance_group(self, actor:"DistributedRayActor", index):
    #     print("start to create new groups")
    #     single_world_size = 2
    #     new_groups = []
    #     master_addr, master_port = self.get_current_addr_port()
    #     group_name = f"group_{index}"
    #     init_method = f"tcp://{master_addr}:{master_port}"
    #     print("New master address", master_addr)
    #     self.init_process_group(
    #             backend="nccl",
    #             init_method=init_method,
    #             world_size=single_world_size,
    #             rank=0,
    #             group_name=group_name
    #         )          
    #     ref=[
    #             actor.init_process_group.remote(
    #                 backend="nccl",
    #                 init_method=init_method,
    #                 world_size=single_world_size,
    #                 rank=j+1,
    #                 group_name=group_name
    #             ) 
    #         ]
    #     ray.get(ref)

    #     new_groups.append(f"group{index}")
    #     return new_groups

    def gather_engine_optimizer(self, rank, world_size, dst): # type: ignore
        print("Group", self.new_group)
        print("Local Rank in Group", self.get_group_rank())
        buffer = io.BytesIO()
        if self.get_rank() == dst:
            gathered_states = [None] * world_size
            gather_object(obj=None, object_gather_list=gathered_states, dst=dst, group=self.new_group)

            all_states = [torch.load(io.BytesIO(state)) for state in gathered_states]
            return all_states

        else:
            state_dict = {
                "engine_state": self.model_engine.state_dict(),
                "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                "config": self.model_engine.config,
            }

            print("Engine_state", self.model_engine.state_dict())
            
            torch.save(state_dict, buffer)
            serialized_state = buffer.getvalue()

            gather_object(obj=serialized_state, dst=dst, group=self.new_group)
    
    def gather_engine_optimizer_ray(self, dst:int, world_size, group_name: str):
        state_dict = {
        "engine_state": self.model_engine.state_dict(),
        "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
        "config": self.model_engine.config,
        }
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        serialized_state = buffer.getvalue()

        state_tensor = torch.ByteTensor(list(serialized_state))#.cuda()
        local_state_size = state_tensor.numel()

        max_size_tensor = torch.tensor([local_state_size], dtype=torch.int64)#.cuda()
        gathered_sizes = [torch.zeros_like(max_size_tensor) for _ in range(world_size)]
        ray.collective.allgather(gathered_sizes, max_size_tensor, group_name=group_name)

        max_size = max([size.item() for size in gathered_sizes])

        # 4. 对齐状态 tensor 的大小，确保每个 tensor 一致
        padded_state_tensor = torch.cat(
            [state_tensor, torch.zeros(max_size - local_state_size, dtype=torch.uint8).cuda()]
        )

        # 5. 通过 allgather 收集所有对齐后的 tensor
        gathered_tensors = [torch.zeros_like(padded_state_tensor) for _ in range(world_size)]
        ray.collective.allgather(gathered_tensors, padded_state_tensor, group_name=group_name)

        # 6. 目标 rank 反序列化数据
        if self.get_rank() == dst:
            all_states = []
            for gathered_tensor, size in zip(gathered_tensors, gathered_sizes):
                serialized = gathered_tensor[:size.item()].cpu().numpy().tobytes()
                all_states.append(torch.load(io.BytesIO(serialized)))
            return all_states

        return None
    

    def print_engine(self):
        state_dict = {
            "engine_state": self.model_engine.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,  
            "config": self.model_engine.config  
            }
            
        print("Engine_state", self.model_engine.state_dict())
    
    def train(self):
        dataset = SimpleDataset(size=1000)
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
        for epoch in range(5):
            self.model_engine.train()
            total_loss = 0.0
            sampler.set_epoch(epoch)
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(dtype=torch.float16, device=self.model_engine.local_rank), labels.to(self.model_engine.local_rank)

                outputs = self.model_engine(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                self.model_engine.backward(loss)
                self.model_engine.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        ray_actor_type (Type[BasePPORole]): PPO model type that this actor group serve on.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
    """

    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type,
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node

        self._initiate_actors(pg, num_gpus_per_actor)

    def _initiate_actors(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        if self._num_gpus_per_node > 1 and pg is None:
            bundles = [
                {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)
            ]
            if self._resources:
                resources_name = list(self._resources.keys())[0]
                for i in range(len(bundles)):
                    bundles[i][resources_name] = self._num_resources_per_node

            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                ),
            ).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(world_size, 0, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        ),
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(
                        num_cpus=num_gpus_per_actor,
                        num_gpus=num_gpus_per_actor,
                        resources=self._resources,
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        """Init model from pretrained checkpoint.

        Returns:
            List: list of remote object refs.
        """
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_fit_actor_model(
        self,
        critic_model_group: "PPORayActorGroup",
        initial_model_group: "PPORayActorGroup",
        reward_model_groups: List["PPORayActorGroup"],
        remote_rm_urls: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List = None,
    ):
        """Train actor model.

        Args:
            critic_model_group (PPORayActorGroup): critic model group.
            initial_model_group (PPORayActorGroup): reference model group.
            reward_model_groups (PPORayActorGroup): reward model groups.
            remote_rm_urls: remote RM APIs.
            reward_fn: reward calculate function, must be specified if using multiple reward models.
            vllm_engines: vllm engines for text generation, if not specified, generate text by actor model directly.

        Returns:
            List: list of remote object refs.
        """
        assert (
            (remote_rm_urls and len(remote_rm_urls) == 1)
            or (reward_model_groups and len(reward_model_groups) == 1)
            or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        critic_actors = critic_model_group._actor_handlers
        initial_actors = initial_model_group._actor_handlers

        refs = []
        # TODO(wuxibin): actor model choose critic/reward/initial model in a
        # round robin fashion, implement more efficient dispatching strategy.
        for i, actor in enumerate(self._actor_handlers):
            critic_actor = critic_actors[i % len(critic_actors)]
            initial_actor = initial_actors[i % len(initial_actors)]

            reward_actors = []
            if not remote_rm_urls:
                for reward_model_group in reward_model_groups:
                    actors = reward_model_group._actor_handlers
                    reward_actors.append(actors[i % len(actors)])

            refs.append(
                actor.fit.remote(
                    critic_model=critic_actor,
                    initial_model=initial_actor,
                    reward_model=reward_actors,
                    remote_rm_url=remote_rm_urls,
                    reward_fn=reward_fn,
                    vllm_engines=vllm_engines,
                    # whether this actor should triger corresponding critic model training
                    critic_train_remote=(i < len(critic_actors)),
                )
            )

        return refs

    def async_save_model(self):
        """Save actor model on rank 0.

        Returns:
            List: list of remote object refs.
        """
        return [actor.save_model.remote() for actor in self._actor_handlers]

    def async_run_method(self, method_name, *args, **kwargs):
        refs = []
        for actor in self._actor_handlers:
            method = getattr(actor, method_name)
            refs.append(method.remote(*args, **kwargs))
        return refs



class DynamicPPORayActorGroup(PPORayActorGroup):
    def __init__(
        self,
        num_nodes,
        num_gpus_per_node,
        ray_actor_type: Type[DistributedRayActor],
        pg: PlacementGroup = None,
        num_gpus_per_actor=1,
        resources: Dict[str, float] = None,
        num_resources_per_node: int = None,
       # world_size: int = None,
        torch_world_size: int = None
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type

        # custom resources, see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        self._resources = resources
        self._num_resources_per_node = num_resources_per_node
        #self.world_size = world_size
        self.torch_world_size = torch_world_size
        self.master_addr = 0
        self.master_port = 0
        self.pgs = []
        self._initiate_actors(pg, num_gpus_per_actor)
        
        
    def _initiate_actors(self, pg, num_gpus_per_actor):
       # self.world_size = self._num_nodes * self._num_gpus_per_node

        # Create placement group if not provided
        # DistributedRayActor(self, rank, world_size, master_addr, master_port):
        print("Create placement group if not provided")
        if pg is None:
            for i in range(self.torch_world_size):
                bundles = [{"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node}]
                print(bundles)
                pg = placement_group(bundles, strategy="STRICT_PACK")
                print(pg)
                ray.get(pg.ready())
                print("pg ready")
                self.pgs.append(pg)

        # Create master actor
        master_actor = self._create_actor(0, num_gpus_per_actor, pg_bundle_index=0)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if self.torch_world_size > 1:
            self.master_addr, self.master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, self.torch_world_size):
                local_rank = rank % self._num_gpus_per_node
                worker_actor = self._create_actor(
                    rank,
                    num_gpus_per_actor
                )
                self._actor_handlers.append(worker_actor)

    def _create_actor(self, rank, num_gpus_per_actor, pg_bundle_index=None):
        # DistributedRayActor(self, rank, world_size, master_addr, master_port):
        print("start to create actor")
        if rank == 0:
            self.master_addr = None
            self.master_port = None
        if self.pgs[rank]:
            return self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pgs[rank])
            ).remote(rank, self.torch_world_size, self.master_addr, self.master_port)
        else:
            return self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(rank, self.torch_world_size, self.master_addr, self.master_port)
        print("create actors")


    def adjust_actor_group_resources(self):
        pass
    

def create_new_instance_groups(
    num_workers_per_group, 
    RayActorGroupList:List[DynamicPPORayActorGroup], 
    new_group:DynamicPPORayActorGroup):
    
    print("start to create new groups")
    single_world_size = len(RayActorGroupList) + 1
    new_groups = []
    for i in range(num_workers_per_group):
        master_addr, master_port = ray.get(new_group._actor_handlers[i].get_current_addr_port.remote())
        group_name = f"group_{i}"
        init_method = f"tcp://{master_addr}:{master_port}"
        print("New master address", master_addr)
        ref = [new_group._actor_handlers[i].init_process_group.remote(
            backend="nccl",
            init_method=init_method,
            world_size=single_world_size,
            rank=0,
            group_name=group_name
        )]
                
        ref.extend([
            actor_group._actor_handlers[i].init_process_group.remote(
                backend="nccl",
                init_method=init_method,
                world_size=single_world_size,
                rank=j+1,
                group_name=group_name
            ) for j, actor_group in enumerate(RayActorGroupList)
        ])
        ray.get(ref)

        new_groups.append(f"group{i}")
    return new_groups
    
# Just create one group
#backend, world_size, rank, group_name
def create_new_instance_group(
    RayActorGroupList:List[DynamicPPORayActorGroup], 
    new_group:DynamicPPORayActorGroup,
    index = 0):
    print("start to create new groups")
    single_world_size = len(RayActorGroupList) + 1
    new_groups = []
    group_name = f"group_{index}"
    ref = [new_group._actor_handlers[index].init_ray_collective_group.remote(
            backend="gloo",
            world_size=single_world_size,
            rank=0,
            group_name=group_name
        )]
        
    ref.extend([
            actor_group._actor_handlers[index].init_ray_collective_group.remote(
            backend="gloo",
            world_size=single_world_size,
            rank=j+1,
            group_name=group_name
            ) for j, actor_group in enumerate(RayActorGroupList)
        ])
    ray.get(ref)

    new_groups.append(f"group{index}")
    return new_groups


def main():
    ray.init(address = "auto")
    
    ds = DynamicPPORayActorGroup(1,1,DistributedRayActor,torch_world_size=2)
    
    model = SimpleModel()
    
    ref = [actor.init_deepspeed.remote(model=model) for actor in ds._actor_handlers]
    
    ray.get(ref)
    
    train_ref = [actor.train.remote() for actor in ds._actor_handlers]
    
    ray.get(train_ref)
    
    # ref = [actor.print_engine.remote() for actor in ds._actor_handlers]
    
    # ray.get(ref)
    
 ########################################################################################   

    ds1 = DynamicPPORayActorGroup(1,1,DistributedRayActor,torch_world_size=2)
    
    init_deepspeed = [actor.init_deepspeed_first_step.remote() for actor in ds1._actor_handlers]
    
    ray.get(init_deepspeed)
    
    #create_new_instance_groups(2, [ds], ds1)
    index = 0
    
    #print("world size1", dist.get_world_size())
    
    group_name = create_new_instance_group([ds], ds1, index)
    # create_new_group = [actor.create_new_instance_groups.remote([ds], i) for i, actor in enumerate(ds1._actor_handlers)]
    #print("world size2", dist.get_world_size())
    # ray.get(create_new_group)
    
    # dst1 = ray.get(ds1._actor_handlers[index].get_rank.remote())
    # dst2 = ray.get(ds1._actor_handlers[1].get_rank.remote())
    
    #ds1._actor_handlers[0].create_new_instance_group.remote(ds._actor_handlers[index],index)
    
    #ray.get(ref)
    
    
    
    # gr1 = ray.get(ds1._actor_handlers[0].get_group_rank.remote())
    # #gr2 = ray.get(ds1._actor_handlers[1].get_group_rank.remote())
    # gr3 = ray.get(ds._actor_handlers[0].get_group_rank.remote())
    # #gr4 = ray.get(ds._actor_handlers[1].get_group_rank.remote())
    
    # print("Group Rank", [gr1, gr3])
    
    # dst = ray.get(ds1._actor_handlers[index].get_rank.remote())
    
    # print("dst rank", dst)
    
    # ref = [ds._actor_handlers[index].gather_engine_optimizer.remote(1, 2, dst)]

    # ref.extend(ds1._actor_handlers[index].gather_engine_optimizer.remote(0, 2, dst))

    ref = [ds._actor_handlers[index].gather_engine_optimizer_ray.remote(0, 2, group_name[0]), ds1._actor_handlers[index].gather_engine_optimizer_ray.remote(0, 2, group_name[0])]
    
    # ref = [actor.gather_engine_optimizer.remote(1, 2, dst) for i, actor in enumerate(ds._actor_handlers)]

    # ref.extend([actor.gather_engine_optimizer.remote(0, 2, dst) for i, actor in enumerate(ds1._actor_handlers)])
    
    results = ray.get(ref)
    
    print("state for Node 1", results[0])
    
    print("state for Node 1", results[1])
    
    
    
    
  #########################################################################  
    
    # # construct communication group
    
    # master_addr, master_port = ray.get(ds._actor_handlers[0].get_master_addr_port.remote())
    
    # ref = [actor.init_process_group.remote(
    #     backend="gloo",
    #     init_method=f"tcp://{master_addr}:{master_port}",
    #     world_size=4,
    #     rank=i,
    #     group_name="deepspeed"
    # ) for i, actor in enumerate(ds._actor_handlers)]
    # ref.extend([actor.init_process_group.remote(
    #     backend="gloo",
    #     init_method=f"tcp://{master_addr}:{master_port}",
    #     world_size=4,
    #     rank=i+2,
    #     group_name="deepspeed"
    # ) for i, actor in enumerate(ds1._actor_handlers)])
    
    # ray.get(ref)
    
    # ref = []
    # for actor, tmp_actor in zip(ds._actor_handlers, ds1._actor_handlers):
    #     engine, optimizer = ray.get(actor.get_engine_optimizer.remote())
    #     ref.append(tmp_actor.init_deepspeed.remote(model=model, model_engine=engine, optimizer=optimizer))
    # ray.get(ref) 
    
    # ref = [actor.train.remote() for actor in ds1._actor_handlers]
    
    # ray.get(ref)
    

    
    print("Group Created")
    

if __name__ == "__main__":
    main()