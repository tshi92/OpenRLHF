import logging
import os
import socket
from typing import Callable, Dict, List, Optional, Type

import ray
import torch
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from openrlhf.utils import DeepspeedStrategy, blending_datasets, get_tokenizer
from openrlhf.datasets import PromptDataset, SFTDataset
from torch.utils.data.sampler import RandomSampler
from openrlhf.models import Actor, get_llm_for_sequence_regression

class DistributedTorchRayActor:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"

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

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output=return_output,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class RewardModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None, packed_seq_lens=None
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(sequences.to(device), attention_mask.to(device), packed_seq_lens=packed_seq_lens)
        return reward.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


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
        ray_actor_type: Type[BasePPORole],
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
    def _initiate_actors(self, pg, num_gpus_per_actor):
        self.world_size = self._num_nodes * self._num_gpus_per_node

        # Create placement group if not provided
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
        self.pg = pg

        # Create master actor
        master_actor = self._create_actor(0, num_gpus_per_actor, pg_bundle_index=0)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if self.world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, self.world_size):
                local_rank = rank % self._num_gpus_per_node
                worker_actor = self._create_actor(
                    rank,
                    num_gpus_per_actor,
                    master_addr=master_addr,
                    master_port=master_port,
                    pg_bundle_index=rank // self._num_gpus_per_node,
                )
                self._actor_handlers.append(worker_actor)

    def _create_actor(self, rank, num_gpus_per_actor, master_addr=None, master_port=None, pg_bundle_index=None):
        if self.pg:
            return self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.pg, placement_group_bundle_index=pg_bundle_index
                ),
            ).remote(self.world_size, rank, rank % self._num_gpus_per_node, master_addr, master_port)
        else:
            return self.ray_actor_type.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                resources=self._resources,
            ).remote(self.world_size, rank, rank % self._num_gpus_per_node, master_addr, master_port)

    def add_actor(self, num_gpus_per_actor=1, *args, **kwargs):
        """
        Dynamically add an actor to the group and update the placement group.
        """
        # Add a new bundle to the placement group
        new_bundle = {"GPU": num_gpus_per_actor, "CPU": num_gpus_per_actor}
        self.pg.bundles.append(new_bundle)
        new_pg = placement_group(self.pg.bundles, strategy="STRICT_SPREAD")
        ray.get(new_pg.ready())

        # Update placement group and add a new actor
        self.pg = new_pg
        new_rank = len(self._actor_handlers)
        new_actor = self._create_actor(rank=new_rank, num_gpus_per_actor=num_gpus_per_actor, pg_bundle_index=new_rank)
        new_actor.init_model_from_pretrained.remote(*args, **kwargs)
        self._actor_handlers.append(new_actor)
        print(f"Added new actor with rank {new_rank}.")

    def remove_actor(self):
        """
        Dynamically remove an actor from the group and update the placement group.
        """
        if len(self._actor_handlers) <= 1:
            print("Cannot remove the last actor.")
            return

        # Remove the last actor
        removed_actor = self._actor_handlers.pop()
        ray.kill(removed_actor)

        # Remove the last bundle from the placement group
        self.pg.bundles.pop()
        new_pg = placement_group(self.pg.bundles, strategy="STRICT_SPREAD")
        ray.get(new_pg.ready())
        self.pg = new_pg
        print(f"Removed an actor. Remaining actors: {len(self._actor_handlers)}.")


    def adjust_actor_group_resources(self):
        pass
    
    
    def prepare_datasets(self, strategy: DeepspeedStrategy): # needs further revision
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        self.prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset, args.micro_rollout_batch_size, True, True, sampler= RandomSampler
        )

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(self.prompts_dataset)))),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None
    
    
    
    def async_fit_actor_model(self,
    critic_model_group: "PPORayActorGroup",
    initial_model_group: "PPORayActorGroup",
    reward_model_groups: List["PPORayActorGroup"],
    strategy: DeepspeedStrategy,
    remote_rm_urls: List[str] = None,
    reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
    vllm_engines: List = None,
    ):
        """
        Train actor model with dynamic resource adjustment.

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

        # Flatten reward actors for easier indexing
        reward_actors = []
        if not remote_rm_urls:
            for reward_model_group in reward_model_groups:
                reward_actors.extend(reward_model_group._actor_handlers)

        # Maximum number of prompts to process
        #### The following code needs further revision
        max_prompts = len(self.prompts_dataloader)

        for prompt_idx in range(max_prompts):
            refs = []
            for i, actor in enumerate(self._actor_handlers):
                critic_actor = critic_actors[i % len(critic_actors)]
                initial_actor = initial_actors[i % len(initial_actors)]

                # Select reward actors based on round-robin
                reward_actor_subset = (
                    [reward_actors[i % len(reward_actors)]] if reward_actors else []
                )

                # Dispatch task to actor
                refs.append(
                    actor.process_single_prompt.remote(
                        critic_model=critic_actor,
                        initial_model=initial_actor,
                        reward_model=reward_actor_subset,
                        remote_rm_url=remote_rm_urls,
                        reward_fn=reward_fn,
                        vllm_engines=vllm_engines,
                    )
                )

            # Wait for current batch to finish processing
            ray.get(refs)

            # Adjust resources dynamically
            self.adjust_actor_group_resources()