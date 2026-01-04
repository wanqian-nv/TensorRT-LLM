import torch
import torch.nn as nn

from tensorrt_llm.llmapi.llm_args import DwdpConfig
from typing import List, Optional, Dict, Tuple
from tensorrt_llm.logger import logger
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._utils import global_mpi_rank
from mpi4py.MPI import COMM_WORLD

from cuda.bindings import runtime as cudart
from cuda.bindings import driver as cuda_driver



# Parameter names to collect handles for
WEIGHT_PARAMS = ['w3_w1_weight', 'w2_weight']
BIAS_PARAMS = ['w3_w1_bias', 'w2_bias']
# Quant scale params vary by quantization method
QUANT_SCALE_PARAMS = [
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
        # Local pointers: param_name -> data_ptr (for verification)
        self.local_ptrs: Dict[str, int] = {}
        # Local offsets: param_name -> offset from allocation base
        # IPC handle points to allocation base, we need offset to get actual tensor data
        self.local_offsets: Dict[str, int] = {}
        # Parameter shapes: param_name -> shape (without expert dim)
        self.param_shapes: Dict[str, torch.Size] = {}
        # Parameter dtypes: param_name -> dtype
        self.param_dtypes: Dict[str, torch.dtype] = {}
        # Peer pointers: (peer_rank, param_name) -> ptr (already adjusted with offset)
        self.peer_ptrs: Dict[Tuple[int, str], int] = {}

    def register_weights(self, module: nn.Module):
        """
        Register weights from a MoE module and create IPC handles.
        
        Called after module.load_weights() completes.
        
        Args:
            module: The MoE module with loaded weights
        """
        # Collect all parameter types
        # Debug: print all parameter names in this module
        all_param_names = [name for name, _ in module.named_parameters(recurse=False)]
        logger.info(f"[DWDP Debug] Module {module.__class__.__name__} has parameters: {all_param_names}")
        
        params_to_register = []
        # Weights (check if present and not None)
        for param_name in WEIGHT_PARAMS:
            if hasattr(module, param_name) and getattr(module, param_name, None) is not None:
                params_to_register.append(param_name)
        # Bias (optional)
        if hasattr(module, 'bias'):
            params_to_register.extend(BIAS_PARAMS)
        # Quant scales (optional, depends on quant method)
        for param_name in QUANT_SCALE_PARAMS:
            if hasattr(module, param_name) and getattr(module, param_name, None) is not None:
                params_to_register.append(param_name)
        logger.info(f"Registering {len(params_to_register)} parameters: {params_to_register}")

        # Register each parameter
        for param_name in params_to_register:
            param = getattr(module, param_name)
            if isinstance(param, nn.Parameter):
                param = param.data
            if param is None:
                continue
            if not param.is_cuda or not param.is_contiguous():
                raise ValueError(f"Parameter {param_name} is not on GPU or is not contiguous")
            self._register_param(param_name, param)
            logger.info(f"Registered parameter {param_name} with shape {param.shape} and dtype {param.dtype}")

    def _register_param(self, param_name: str, param: torch.Tensor):
        # Get IPC handle - note: handle points to the CUDA allocation base, not tensor's data_ptr
        tensor_ptr = param.data_ptr()
        err, handle = cudart.cudaIpcGetMemHandle(tensor_ptr)
        check_cuda_error(err, f"get handle for {param_name}")
        
        # Get allocation base address using Driver API cuMemGetAddressRange
        # This returns the actual base address and size of the CUDA allocation
        # cudaPointerGetAttributes.devicePointer returns the input pointer, not base!
        err, alloc_base, alloc_size = cuda_driver.cuMemGetAddressRange(tensor_ptr)
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemGetAddressRange failed for {param_name}: {err}")
        
        # Calculate offset from allocation base
        # Convert CUdeviceptr to int for arithmetic
        offset = tensor_ptr - int(alloc_base)
        
        self.local_ipc_handles[param_name] = bytes(handle.reserved)
        self.local_ptrs[param_name] = tensor_ptr
        self.local_offsets[param_name] = offset
        self.param_shapes[param_name] = param.shape[1:]
        self.param_dtypes[param_name] = param.dtype
        
        if offset != 0:
            logger.info(f"[DWDP] Parameter {param_name} has non-zero offset: {offset} base: {hex(int(alloc_base))} bytes from allocation base (alloc_size={alloc_size})")
            # Read data at alloc_base for debugging (need cudaMemcpy since we only have raw pointer)
            debug_tensor = torch.zeros(1, dtype=param.dtype, device='cuda')
            err, = cudart.cudaMemcpy(
                debug_tensor.data_ptr(),
                int(alloc_base),
                debug_tensor.numel() * debug_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            )
            logger.info(f"[DWDP] Data at alloc_base: {debug_tensor.cpu().tolist()}")
            # Directly read actual tensor data (no cudaMemcpy needed)
            logger.info(f"[DWDP] Data at tensor_ptr: {param.flatten()[:10].cpu().tolist()}")
        logger.info(f"Registered parameter {param_name} with shape {param.shape} and dtype {param.dtype}")

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
                'offsets': collector.local_offsets,  # Include offsets for IPC pointer adjustment
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
                peer_offsets = ipc_collector['offsets']
                for param_name, handle_bytes in ipc_collector['handles'].items():
                    logger.info(f"Opening handle for param {param_name} from rank {peer_rank} layer {layer_idx}")
                    # Reconstruct and open handle
                    handle = cudart.cudaIpcMemHandle_t()
                    handle.reserved = list(handle_bytes)
                    
                    err, base_ptr = cudart.cudaIpcOpenMemHandle(
                        handle,
                        cudart.cudaIpcMemLazyEnablePeerAccess
                    )
                    check_cuda_error(err, f"open handle rank={peer_rank}")
                    
                    # Apply offset to get actual tensor pointer
                    # IPC handle points to allocation base, offset gives us the tensor location
                    offset = peer_offsets[param_name]
                    actual_ptr = base_ptr + offset
                    if offset != 0:
                        logger.info(f"[DWDP] Applying offset {offset} base: {hex(base_ptr)} for {param_name} from rank {peer_rank}")
                    collector.peer_ptrs[(peer_rank, param_name)] = actual_ptr

    def verify_ipc_communication(self, num_elements: int = 10):


        x = torch.empty(1024 * 1024, device='cuda', dtype=torch.uint8)
        real_base = x.data_ptr()

        # 2. 取中间的一个切片
        y = x[512:]
        offset_ptr = y.data_ptr()

        # 3. 查询 offset_ptr 的属性
        err, attr = cudart.cudaPointerGetAttributes(offset_ptr)
        err, base_ptr, size = cuda_driver.cuMemGetAddressRange(offset_ptr)

        logger.info(f"x: {hex(real_base)} y: {hex(offset_ptr)}")
        logger.info(f"devicePointer: {hex(attr.devicePointer)}")
        logger.info(f"cuMemGetAddressRange: {hex(int(base_ptr))}")
        logger.info(f"Is same? {real_base == attr.devicePointer}")
        logger.info(f"Is same? {real_base == base_ptr}")

        logger.info(f"[DWDP] Rank {self.rank}: Starting IPC communication verification with {num_elements} elements")
            
        collector = self.ipc_collectors[0]
        layer_idx = 0
        logger.info(f"[DWDP] Rank {self.rank}: Verifying layer {layer_idx}")
        
        # Step 1: Read local weight samples for each parameter
        local_samples = {}  # param_name -> tensor (on CPU)
        for param_name, local_ptr in collector.local_ptrs.items():
            param_shape = collector.param_shapes[param_name]
            param_dtype = collector.param_dtypes[param_name]
            
            total_elements = param_shape.numel()
            copy_elements = min(num_elements, total_elements)
            
            # Create buffer and read local data
            local_tensor = torch.zeros(copy_elements, dtype=param_dtype, device='cuda')
            bytes_to_copy = copy_elements * local_tensor.element_size()
            err, = cudart.cudaMemcpy(
                local_tensor.data_ptr(),
                local_ptr,
                bytes_to_copy,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            )
            check_cuda_error(err, f"rank {self.rank} read local data for {param_name}")
            local_samples[param_name] = local_tensor.cpu().numpy().tolist()
            logger.info(f"Local sample for {param_name}: dtype: {param_dtype} bytes: {bytes_to_copy} sample: {local_samples[param_name]}")
        
        # Synchronize before allgather
        torch.cuda.synchronize()
        self.dwdp_group.barrier()
        
        # Step 2: Exchange local samples via MPI allgather (ground truth)
        local_data = {
            'rank': self.rank,
            'samples': local_samples,
        }
        all_samples = self.dwdp_group.allgather(local_data)
        
        # Build ground truth map: (peer_rank, param_name) -> expected_values
        ground_truth = {}
        for peer_data in all_samples:
            peer_rank = peer_data['rank']
            if peer_rank != self.rank:
                for param_name, values in peer_data['samples'].items():
                    ground_truth[(peer_rank, param_name)] = values
        
        logger.info(f"[DWDP] Rank {self.rank}: Collected ground truth from {len(all_samples)-1} peers")
        
        # Step 3 & 4: Read via IPC and compare with ground truth
        for (peer_rank, param_name), peer_ptr in collector.peer_ptrs.items():
            param_shape = collector.param_shapes[param_name]
            param_dtype = collector.param_dtypes[param_name]
            
            total_elements = param_shape.numel()
            copy_elements = min(num_elements, total_elements)
            
            # Read via IPC
            recv_tensor = torch.zeros(copy_elements, dtype=param_dtype, device='cuda')
            bytes_to_copy = copy_elements * recv_tensor.element_size()
            err, = cudart.cudaMemcpy(
                recv_tensor.data_ptr(),
                peer_ptr,
                bytes_to_copy,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            )
            check_cuda_error(err, f"rank {self.rank} IPC read from rank {peer_rank}")
            
            ipc_values = recv_tensor.cpu().numpy().tolist()
            expected_values = ground_truth.get((peer_rank, param_name), None)
            
            if expected_values is None:
                logger.error(
                    f"[DWDP] ❌ Rank {self.rank}: No ground truth for rank {peer_rank}, param={param_name}"
                )
                continue
            
            # Compare IPC values with ground truth
            if ipc_values == expected_values:
                logger.info(
                    f"[DWDP] ✅ Rank {self.rank}: IPC verified from rank {peer_rank}, "
                    f"layer={layer_idx}, param={param_name}, "
                    f"values match! sample={ipc_values[:3]}..."
                )
            else:
                logger.error(
                    f"[DWDP] ❌ Rank {self.rank}: IPC MISMATCH from rank {peer_rank}, "
                    f"layer={layer_idx}, param={param_name}, "
                    f"expected={expected_values[:3]}..., got={ipc_values[:3]}..."
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
        
