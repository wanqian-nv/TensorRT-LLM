from tensorrt_llm.llmapi.llm_args import DwdpConfig
from typing import List, Optional, Dict, Tuple
from tensorrt_llm.logger import logger
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._utils import global_mpi_rank
from mpi4py.MPI import COMM_WORLD

import torch.distributed as dist

_global_dwdp_manager: Optional["DwdpManager"] = None


def set_global_dwdp_manager(manager: "DwdpManager"):
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


class DwdpLayerHandleCollector:
    """
    Dwdp Layer Handle Collector for IPC handle coordination and prefetch buffer management.
    """
    
    def __init__(
        self,
    ):
        pass

    def register_weights(self):
        """
        Register weights for DWDP.
        """
        pass

    def get_peer_ptr(self, peer_rank: int, param_name: str) -> int:
        """
        Get peer pointer for a specific parameter.
        """
        pass


class DwdpPrefetchBuffer:
    """
    Dwdp Prefetch Buffer for prefetching remote expert weights.
    """
    pass


class DwdpManager:
    """
    Dwdp Manager for IPC handle coordination and prefetch buffer management.
    
    This manager:
    - Tracks IPC handles for all MoE layers across Context workers
    - Manages double-buffered prefetch buffers for remote expert weights
    - Provides expert tensor routing (local vs. prefetched)
    
    """
    
    def __init__(
        self,
        config: DwdpConfig,
        dist: Optional[object] = None,
    ):

        self.config = config
        self.dist = dist
        self.dwdp_size = config.dwdp_size
        self.experts_per_worker = config.experts_per_worker

        self._init_dwdp_group()
        
        # Per-layer IPC handle collectors (indexed by layer_idx)
        self.ipc_collectors: List[DwdpLayerHandleCollector] = []
        
        # Prefetch buffer (initialized later in create_py_executor)
        self.prefetch_buffer: Optional[DwdpPrefetchBuffer] = None

        set_global_dwdp_manager(self)

    def _init_dwdp_group(self):

        assert isinstance(self.dist, MPIDist), "Dwdp Communicator requires MPI backend"

        self.rank = global_mpi_rank()
        ranks = list(range(self.dwdp_size))
        new_group = COMM_WORLD.group.Incl(ranks)
        self.dwdp_group = COMM_WORLD.Create_group(new_group)
        self.start_expert_id = self.rank * self.experts_per_worker
        logger.info(f"Rank {self.rank} initialized Dwdp Group with ranks: {ranks} from MPI_COMM_WORLD")

    def is_enabled(self) -> bool:
        return self.config.enabled and self.dwdp_size > 1
        
    # def add_layer(
    #     self,
    #     layer_idx: int,
    #     num_experts: int,
    #     expert_size_per_partition: int,
    # ) -> "DwdpLayerHandleCollector":
    #     """
    #     Add a new layer IPC handle collector.
        
    #     Called from CuteDslFusedMoE.__init__() during model construction.
        
    #     Args:
    #         layer_idx: Layer index
    #         num_experts: Total number of experts
    #         expert_size_per_partition: Experts per rank
            
    #     Returns:
    #         DWDPLayerHandleCollector for this layer
    #     """
    #     collector = DWDPLayerHandleCollector(
    #         manager=self,
    #         layer_idx=layer_idx,
    #         num_experts=num_experts,
    #         expert_size_per_partition=expert_size_per_partition,
    #     )
    #     self.ipc_collectors.append(collector)
    #     return collector
        
    # def exchange_all_handles(self):
    #     """
    #     Exchange IPC handles with peer Context workers via DWDP Group AllGather.
        
    #     Called after all weights are loaded, before creating prefetch buffer.
    #     """
    #     import torch.distributed as dist
            
    #     # Collect all local handles with explicit worker info
    #     local_data = {
    #         'worker_id': self.worker_id,
    #         'expert_start_idx': self.expert_start_idx,
    #         'layers': {}
    #     }
    #     for collector in self.ipc_collectors:
    #         local_data['layers'][collector.layer_idx] = {
    #             'handles': collector.local_ipc_handles,
    #             'sizes': collector.param_sizes,
    #             'shapes': collector.param_shapes,
    #             'dtypes': {k: str(v) for k, v in collector.param_dtypes.items()},
    #         }
            
    #     # AllGather from all Context workers in DWDP group
    #     all_data = [None] * self.num_ctx_workers
    #     dist.all_gather_object(all_data, local_data, group=self.dwdp_group)
        
    #     # Open handles from peer workers
    #     for peer_data in all_data:
    #         peer_worker_id = peer_data['worker_id']
    #         if peer_worker_id == self.worker_id:
    #             continue
                
    #         for layer_idx, layer_data in peer_data['layers'].items():
    #             if layer_idx >= len(self.ipc_collectors):
    #                 continue
                    
    #             collector = self.ipc_collectors[layer_idx]
                    
    #             for param_name, handle_bytes in layer_data['handles'].items():
    #                 # Reconstruct and open handle
    #                 handle = cudart.cudaIpcMemHandle_t()
    #                 handle.reserved = list(handle_bytes)
                    
    #                 err, ptr = cudart.cudaIpcOpenMemHandle(
    #                     handle,
    #                     cudart.cudaIpcMemLazyEnablePeerAccess
    #                 )
    #                 self._check_cuda_error(err, f"open handle worker={peer_worker_id}")
                    
    #                 # Store with worker_id as key
    #                 collector.peer_ptrs[(peer_worker_id, param_name)] = ptr
        
    # def get_collector(self, layer_idx: int) -> "DWDPLayerHandleCollector":
    #     """Get collector for a specific layer (direct index access)."""
    #     return self.ipc_collectors[layer_idx]
        
    # def get_param_info_for_buffer(self) -> Tuple[Dict[str, torch.Size], Dict[str, torch.dtype]]:
    #     """
    #     Get parameter shapes and dtypes for prefetch buffer allocation.
        
    #     Returns:
    #         (param_shapes, param_dtypes) from the first layer collector
    #     """
    #     if not self.ipc_collectors:
    #         return {}, {}
    #     return self.ipc_collectors[0].param_shapes, self.ipc_collectors[0].param_dtypes
        
    # def initialize_prefetch_buffer(
    #     self,
    #     num_remote_experts: int,
    #     param_shapes: Dict[str, torch.Size],
    #     param_dtypes: Dict[str, torch.dtype],
    #     device: torch.device,
    #     num_layers: int,
    #     remote_ranks: List[int],
    # ):
    #     """
    #     Initialize the prefetch buffer.
        
    #     Called in create_py_executor() after model loading.
        
    #     Args:
    #         num_remote_experts: Total experts from remote ranks
    #         param_shapes: Shape of each param (without expert dim)
    #         param_dtypes: Dtype of each param
    #         device: Device to allocate on
    #         num_layers: Number of MoE layers
    #         remote_ranks: List of remote DWDP rank IDs to prefetch from
    #     """
    #     self.remote_ranks = remote_ranks
        
    #     # Build rank -> buffer slot mapping
    #     # Buffer layout: [rank0_experts..., rank1_experts..., ...]
    #     self.rank_to_buffer_slot: Dict[int, int] = {}
    #     for slot_idx, r in enumerate(remote_ranks):
    #         self.rank_to_buffer_slot[r] = slot_idx
            
    #     self.prefetch_buffer = DWDPPrefetchBuffer(
    #         num_remote_experts=num_remote_experts,
    #         param_shapes=param_shapes,
    #         param_dtypes=param_dtypes,
    #         device=device,
    #         num_layers=num_layers,
    #         experts_per_rank=self.experts_per_rank,
    #     )
        
    # # -------------------------------------------------------------------------
    # # Prefetch Operations
    # # -------------------------------------------------------------------------
    
    # def prefetch_layer(self, layer_idx: int):
    #     """
    #     Prefetch all remote experts for a specific layer.
    #     Copies entire rank's experts in one cudaMemcpyAsync call per param.
        
    #     Args:
    #         layer_idx: Layer to prefetch
    #     """
    #     if self.prefetch_buffer is None:
    #         return
            
    #     collector = self.get_collector(layer_idx)
    #     write_buffer = self.prefetch_buffer.get_write_buffer_for_layer(layer_idx)
        
    #     with torch.cuda.stream(self.prefetch_buffer.prefetch_stream):
    #         for remote_rank in self.remote_ranks:
    #             buffer_slot = self.rank_to_buffer_slot[remote_rank]
                
    #             # Copy all experts from this rank in one call per param
    #             for param_name in write_buffer.keys():
    #                 # Source: peer GPU (entire param tensor for all experts)
    #                 src_ptr = collector.get_peer_ptr(remote_rank, param_name)
    #                 total_size = collector.param_sizes[param_name]
                    
    #                 # Destination: contiguous slot in prefetch buffer
    #                 dst_ptr = self.prefetch_buffer.get_rank_slot_ptr(
    #                     layer_idx, buffer_slot, param_name
    #                 )
                    
    #                 err = cudart.cudaMemcpyAsync(
    #                     dst_ptr,
    #                     src_ptr,
    #                     total_size,
    #                     cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
    #                     self.prefetch_buffer.prefetch_stream.cuda_stream,
    #                 )
    #                 self._check_cuda_error(err, f"prefetch layer {layer_idx} rank {remote_rank}")
                    
    #         # Record completion event
    #         self.prefetch_buffer.record_event(layer_idx)
    
    # def get_expert_tensor(
    #     self, 
    #     layer_idx: int, 
    #     expert_id: int, 
    #     param_name: str,
    #     local_param: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Get tensor for a specific expert's parameter.
    #     Automatically routes to local weight or prefetch buffer based on expert_id.
        
    #     Args:
    #         layer_idx: Layer index
    #         expert_id: Global expert ID
    #         param_name: Parameter name (e.g., 'w3_w1_weight')
    #         local_param: Local parameter tensor for this rank
            
    #     Returns:
    #         Tensor view of the expert's parameter
    #     """
    #     # Determine which rank owns this expert
    #     owner_rank = expert_id // self.experts_per_rank
    #     local_idx = expert_id % self.experts_per_rank
        
    #     if owner_rank == self.rank:
    #         # Local expert - get from local param
    #         return local_param[local_idx]
    #     else:
    #         # Remote expert - get from prefetch buffer
    #         buffer_slot = self.rank_to_buffer_slot[owner_rank]
    #         return self.prefetch_buffer.get_expert_tensor(
    #             layer_idx, buffer_slot, local_idx, param_name
    #         )
            
    # def wait_layer(self, layer_idx: int):
    #     """Wait for a layer's prefetch to complete."""
    #     if self.prefetch_buffer is None:
    #         return
    #     self.prefetch_buffer.wait_event(layer_idx)
        
    # def prefetch_all_layers(self):
    #     """Prefetch all layers asynchronously."""
    #     for layer_idx in range(len(self.ipc_collectors)):
    #         self.prefetch_layer(layer_idx)
        
    # def cleanup(self):
    #     """Clean up all IPC handles."""
    #     for collector in self.ipc_collectors:
    #         collector.cleanup()
        
    # @staticmethod
    # def _check_cuda_error(err, context: str = ""):
    #     """Check CUDA error."""
    #     if err != cudart.cudaError_t.cudaSuccess:
    #         raise RuntimeError(f"CUDA error in {context}: {err}")