import torch
import torch.nn as nn

from tensorrt_llm.llmapi.llm_args import DwdpConfig
from typing import List, Optional, Dict, Tuple
from tensorrt_llm.logger import logger
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._utils import global_mpi_rank
from mpi4py.MPI import COMM_WORLD
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


# Parameter names to collect handles for
WEIGHT_PARAMS = ['w3_w1_weight', 'w2_weight']
BIAS_PARAMS = ['w3_w1_bias', 'w2_bias']
# Quant scale params vary by quantization method
QUANT_SCALE_PARAMS = [
    'w3_w1_weight_scaling_factor', 'w2_weight_scaling_factor',  # FP8
    'w3_w1_weight_scale', 'w2_weight_scale',  # NVFP4/MXFP4
    'fc31_alpha', 'fc2_alpha',  # NVFP4 alpha
]


_global_dwdp_manager: Optional["DwdpManager"] = None


def set_global_dwdp_manager(manager: "DwdpManager"):
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


def check_cuda_error(err, context: str = ""):
    """Check CUDA error."""
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error in {context}: {err}")


class DwdpLayerHandleCollector:
    """
    Dwdp Layer Handle Collector for IPC handle coordination and prefetch buffer management.
    """
    
    def __init__(
        self,
        layer_idx: int,
    ):

        self.layer_idx = layer_idx

        # Local IPC handles: param_name -> handle_bytes
        self.local_ipc_handles: Dict[str, bytes] = {}
        # Parameter shapes: param_name -> shape (without expert dim)
        self.param_shapes: Dict[str, torch.Size] = {}
        # Parameter dtypes: param_name -> dtype
        self.param_dtypes: Dict[str, torch.dtype] = {}
        # Peer pointers: (peer_rank, param_name) -> ptr
        self.peer_ptrs: Dict[Tuple[int, str], int] = {}

    def register_weights(self, module: nn.Module):
        """
        Register weights from a MoE module and create IPC handles.
        
        Called after module.load_weights() completes.
        
        Args:
            module: The MoE module with loaded weights
        """
        # Collect all parameter types
        params_to_register = []
        # Weights (always present)
        params_to_register.extend(WEIGHT_PARAMS)
        # Bias (optional)
        if hasattr(module, 'bias'):
            params_to_register.extend(BIAS_PARAMS)
        # Quant scales (optional, depends on quant method)
        for param_name in QUANT_SCALE_PARAMS:
            if hasattr(module, param_name):
                params_to_register.append(param_name)
                
        # Register each parameter
        for param_name in params_to_register:
            param = getattr(module, param_name)
            if isinstance(param, nn.Parameter):
                param = param.data
            if not param.is_cuda or not param.is_contiguous():
                raise ValueError(f"Parameter {param_name} is not on GPU or is not contiguous")
            self._register_param(param_name, param)
            logger.info(f"Registered parameter {param_name} with shape {param.shape} and dtype {param.dtype}")

    def _register_param(self, param_name: str, param: torch.Tensor):
        # Get IPC handle
        err, handle = cudart.cudaIpcGetMemHandle(param.data_ptr())
        check_cuda_error(err, f"get handle for {param_name}")
        
        self.local_ipc_handles[param_name] = bytes(handle.reserved)
        self.param_shapes[param_name] = param.shape[1:]
        self.param_dtypes[param_name] = param.dtype

    def get_peer_ptr(self, peer_rank: int, param_name: str) -> int:
        """Get pointer to parameter on peer rank."""
        return self.peer_ptrs[(peer_rank, param_name)]
        
    def cleanup(self):
        """Clean up peer handles."""
        for _, ptr in self.peer_ptrs.items():
            cudart.cudaIpcCloseMemHandle(ptr)
        self.peer_ptrs.clear()


class DwdpPrefetchBuffer:
    """
    Ping-pong buffer for expert weight prefetching.
    
    Buffer Selection Strategy:
    - Even layers (0, 2, 4, ...) use buffer[0]
    - Odd layers (1, 3, 5, ...) use buffer[1]
    - This ensures layer N-1's prefetch doesn't overwrite layer N's data
    
    Synchronization Strategy:
    - prefetch_events[buffer_idx][layer_idx]: Recorded when prefetch completes
      Waited by forward() before using prefetched data
    - compute_events[buffer_idx][layer_idx]: Recorded when forward() completes
      Waited by next prefetch before overwriting buffer
    
    Buffer Layout (organized by rank slot):
    - buffers[buffer_idx][param_name] = Tensor[num_remote_experts, ...]
    - Data arranged as: [slot0_experts..., slot1_experts..., ...]
    - Each slot contains experts_per_rank experts from one remote rank
    """
    def __init__(
        self,
        dwdp_size: int,
        experts_per_worker: int,
        num_layers: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
    ):

        self.num_peer_ranks = dwdp_size - 1
        self.experts_per_worker = experts_per_worker
        self.num_prefetch_experts = self.num_peer_ranks * self.experts_per_worker
        self.num_layers = num_layers
        self.num_buffers = 2  # Ping-pong
        
        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes
        
        self.device = torch.cuda.current_device()
        
        self.buffers: List[Dict[str, torch.Tensor]] = []
        
        for _ in range(self.num_buffers):
            buffer = {}
            for param_name, shape in param_shapes.items():
                dtype = param_dtypes[param_name]
                buffer_shape = (self.num_prefetch_experts,) + tuple(shape)
                buffer[param_name] = torch.empty(
                    buffer_shape,
                    dtype=dtype,
                    device=self.device,
                )
            self.buffers.append(buffer)
            
        self.prefetch_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(num_layers//self.num_buffers)]
            for _ in range(self.num_buffers)
        ]
        self.compute_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(num_layers//self.num_buffers)]
            for _ in range(self.num_buffers)
        ]
        self.prefetch_stream = torch.cuda.Stream(device=self.device)

    def initialize_compute_events(self):
        for buffer_idx in range(self.num_buffers):
            self.compute_events[buffer_idx][0].record(torch.cuda.current_stream())
    
    def record_prefetch_event(self, layer_idx: int):
        self.prefetch_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers].record(self.prefetch_stream)
    
    def record_compute_event(self, layer_idx: int):
        self.compute_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers].record(torch.cuda.current_stream())
        
    def wait_prefetch_event(self, layer_idx: int):
        torch.cuda.current_stream().wait_event(self.prefetch_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers])
    
    def wait_compute_event(self, layer_idx: int):
        self.prefetch_stream.wait_event(self.compute_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers])


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
        
    def add_layer(
        self,
        layer_idx: int,
    ) -> "DwdpLayerHandleCollector":
        """
        Add a new layer IPC handle collector.
        
        Called from CuteDslFusedMoE.__init__() during model construction.
        """
        collector = DwdpLayerHandleCollector(
            layer_idx=layer_idx
        )
        self.ipc_collectors.append(collector)
        return collector
        
    def exchange_all_handles(self):
        """
        Exchange IPC handles with peer Context workers via Dwdp Group AllGather.
        
        Called after all weights are loaded, before creating prefetch buffer.
        """
            
        # Collect all local handles with explicit worker info
        local_data = {
            'rank': self.rank,
            'expert_start_id': self.start_expert_id,
            'ipc_collectors': [],
        }
        for collector in self.ipc_collectors:
            local_data['ipc_collectors'].append({
                'layer_idx': collector.layer_idx,
                'handles': collector.local_ipc_handles,
            })
            
        # AllGather from all Context workers in DWDP group
        all_data = self.dwdp_group.allgather(local_data)
        
        # Open handles from peer workers
        for peer_data in all_data:
            peer_rank = peer_data['rank']
            if peer_rank == self.rank:
                continue
            for layer_idx, ipc_collector in enumerate(peer_data['ipc_collectors']):
                collector = self.ipc_collectors[layer_idx]
                for param_name, handle_bytes in ipc_collector['handles'].items():
                    # Reconstruct and open handle
                    handle = cudart.cudaIpcMemHandle_t()
                    handle.reserved = list(handle_bytes)
                    
                    err, ptr = cudart.cudaIpcOpenMemHandle(
                        handle,
                        cudart.cudaIpcMemLazyEnablePeerAccess
                    )
                    check_cuda_error(err, f"open handle rank={peer_rank}")
                    collector.peer_ptrs[(peer_rank, param_name)] = ptr

    def verify_ipc_communication(self, num_elements: int = 10):

        logger.info(f"[DWDP] Rank {self.rank}: Starting IPC communication verification with {num_elements} elements")

        for layer_idx, collector in enumerate(self.ipc_collectors):
            logger.info(f"[DWDP] Rank {self.rank}: Verifying layer {layer_idx}")
            
            for param_name, local_ptr in collector.local_ipc_handles.items():
                param_shape = collector.param_shapes[param_name]
                param_dtype = collector.param_dtypes[param_name]
                
                # Calculate actual number of elements to copy
                total_elements = param_shape.numel()
                copy_elements = min(num_elements, total_elements)
                
                # Create a local tensor for verification
                local_tensor = torch.zeros(copy_elements, dtype=param_dtype, device='cuda')
                local_tensor.fill_(float(self.rank))  # Fill with rank number for identification
                
                # Copy local tensor data to local IPC memory
                src_ptr = local_tensor.data_ptr()
                bytes_to_copy = copy_elements * local_tensor.element_size()
                err, = cudart.cudaMemcpy(
                    local_ptr,
                    src_ptr,
                    bytes_to_copy,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
                )
                check_cuda_error(err, f"rank {self.rank} local memcpy layer={layer_idx} param={param_name}")
                
                logger.info(
                    f"[DWDP] Rank {self.rank}: Written {copy_elements} elements "
                    f"(value={float(self.rank)}) to local ptr for {param_name}"
                )
            
            # Synchronize before reading from peers
            torch.cuda.synchronize()
            self.dwdp_group.barrier()
            
            # Now read from peer ranks
            for (peer_rank, param_name), peer_ptr in collector.peer_ptrs.items():
                param_shape = collector.param_shapes[param_name]
                param_dtype = collector.param_dtypes[param_name]
                
                total_elements = param_shape.numel()
                copy_elements = min(num_elements, total_elements)
                
                recv_tensor = torch.zeros(copy_elements, dtype=param_dtype, device='cuda')
                
                dst_ptr = recv_tensor.data_ptr()
                bytes_to_copy = copy_elements * recv_tensor.element_size()
                err, = cudart.cudaMemcpy(
                    dst_ptr,
                    peer_ptr,
                    bytes_to_copy,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
                )
                check_cuda_error(err, f"rank {self.rank} peer memcpy from rank {peer_rank}")
                
                expected_value = float(peer_rank)
                actual_values = recv_tensor.cpu().numpy()
                
                if torch.allclose(recv_tensor, torch.full_like(recv_tensor, expected_value)):
                    logger.info(
                        f"[DWDP] ✅ Rank {self.rank}: Successfully verified IPC from rank {peer_rank}, "
                        f"layer={layer_idx}, param={param_name}, "
                        f"received values={actual_values[:5]}... (expected {expected_value})"
                    )
                else:
                    logger.error(
                        f"[DWDP] ❌ Rank {self.rank}: FAILED IPC verification from rank {peer_rank}, "
                        f"layer={layer_idx}, param={param_name}, "
                        f"expected {expected_value}, got {actual_values[:5]}..."
                    )
            
            self.dwdp_group.barrier()
        
        logger.info(f"[DWDP] Rank {self.rank}: IPC communication verification completed")

    def initialize_prefetch_buffer(self):
        """
        Initialize the prefetch buffer.
        
        Called in create_py_executor() after model loading.
        """
        self.prefetch_buffer = DwdpPrefetchBuffer(
            dwdp_size=self.dwdp_size,
            experts_per_worker=self.experts_per_worker,
            num_layers=len(self.ipc_collectors),
            param_shapes=self.ipc_collectors[0].param_shapes,
            param_dtypes=self.ipc_collectors[0].param_dtypes,
        )
        self.prefetch_buffer.initialize_compute_events()
        
