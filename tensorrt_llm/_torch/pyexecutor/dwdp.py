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
from tensorrt_llm._utils import nvtx_range



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
        # Local samples for verification: param_name -> list of sampled values
        # Sample at positions [0, 8, 16, ..., 56] * expert_size
        self.local_samples: Dict[str, List[float]] = {}

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
        
        # Sample param for verification: sample at expert positions [0, 8, 16, ..., 56]
        # Each sample takes the first element of that expert's data
        num_experts = param.shape[0]
        sample_positions = list(range(0, min(64, num_experts), 8))  # [0, 8, 16, 24, 32, 40, 48, 56]
        samples = []
        for expert_idx in sample_positions:
            sample_value = param[expert_idx].flatten()[0].item()
            samples.append(sample_value)
        self.local_samples[param_name] = samples
        logger.info(f"[DWDP Sample] {param_name}: sampled {len(samples)} experts at positions {sample_positions}, values={samples[:4]}...")
        
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
        num_prefetch_experts: int,
        num_layers: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
    ):

        self.dwdp_size = dwdp_size
        self.num_prefetch_experts = num_prefetch_experts
        self.experts_per_worker = experts_per_worker
        self.num_experts = self.num_prefetch_experts * (self.dwdp_size - 1) + self.experts_per_worker
        self.num_layers = num_layers
        self.num_buffers = 2  # Ping-pong
        
        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes
        
        self.device = torch.cuda.current_device()
        
        self.buffers: List[Dict[str, torch.Tensor]] = []

        logger.info(f"[DWDP] Initializing prefetch buffer with {self.num_experts} experts")
        
        # Use num_experts size so we can index by global expert_id
        for _ in range(self.num_buffers):
            buffer = {}
            for param_name, shape in param_shapes.items():
                dtype = param_dtypes[param_name]
                buffer_shape = (self.num_experts,) + tuple(shape)
                buffer[param_name] = torch.empty(
                    buffer_shape,
                    dtype=dtype,
                    device=self.device,
                )
            self.buffers.append(buffer)
            
        # Use (num_layers + 3) to account for dense layers before MoE layers
        # e.g., DeepSeek has 3 dense layers, so global layer_idx starts from 3
        self.max_layer_idx = num_layers + 3
        self.prefetch_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(self.max_layer_idx//self.num_buffers + 1)]
            for _ in range(self.num_buffers)
        ]
        self.compute_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(self.max_layer_idx//self.num_buffers + 1)]
            for _ in range(self.num_buffers)
        ]
        self.prefetch_stream = torch.cuda.Stream(device=self.device)

    def initialize_compute_events(self):
        for buffer_idx in range(self.num_buffers):
            self.compute_events[buffer_idx][0].record(torch.cuda.current_stream())
    
    def record_prefetch_event(self, layer_idx: int):
        logger.info(f"[DWDP] Record prefetch event for layer {layer_idx} buffer_idx: {layer_idx % self.num_buffers} event_idx: {layer_idx // self.num_buffers}")
        self.prefetch_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers].record(self.prefetch_stream)
    
    def record_compute_event(self, layer_idx: int):
        logger.info(f"[DWDP] Record compute event for layer {layer_idx} buffer_idx: {layer_idx % self.num_buffers} event_idx: {layer_idx // self.num_buffers}")
        self.compute_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers].record(torch.cuda.current_stream())
        
    def wait_prefetch_event(self, layer_idx: int):
        logger.info(f"[DWDP] Wait prefetch event for layer {layer_idx} buffer_idx: {layer_idx % self.num_buffers} event_idx: {layer_idx // self.num_buffers}")
        torch.cuda.current_stream().wait_event(self.prefetch_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers])
    
    def wait_compute_event(self, layer_idx: int):
        logger.info(f"[DWDP] Wait compute event for layer {layer_idx} buffer_idx: {layer_idx % self.num_buffers} event_idx: {layer_idx // self.num_buffers}")
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

        # Peer expert ranges: (peer_rank, (start_expert_id, end_expert_id))
        self.peer_expert_ranges: Dict[int, Tuple[int, int]] = {}
        # All samples for verification: (layer_idx, param_name) -> {rank: samples}
        # Samples are taken at expert positions [0, 8, 16, ..., 56]
        self.all_samples: Dict[Tuple[int, str], Dict[int, List[float]]] = {}

        self.num_prefetch_experts = config.num_prefetch_experts
        self.start_expert_id = self.num_prefetch_experts * self.rank
        self.end_expert_id = self.start_expert_id + self.experts_per_worker
        
        logger.info(f"Rank {self.rank} expert range: [{self.start_expert_id}, {self.end_expert_id}), prefetch={self.num_prefetch_experts}")

        set_global_dwdp_manager(self)

    def _init_dwdp_group(self):

        assert isinstance(self.dist, MPIDist), "Dwdp Communicator requires MPI backend"

        self.rank = global_mpi_rank()
        ranks = list(range(self.dwdp_size))
        new_group = COMM_WORLD.group.Incl(ranks)
        self.dwdp_group = COMM_WORLD.Create_group(new_group)
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
            'expert_end_id': self.end_expert_id,
            'ipc_collectors': [],
        }
        for collector in self.ipc_collectors:
            local_data['ipc_collectors'].append({
                'layer_idx': collector.layer_idx,
                'handles': collector.local_ipc_handles,
                'offsets': collector.local_offsets,  # Include offsets for IPC pointer adjustment
                'samples': collector.local_samples,  # Include samples for verification
            })
            
        # AllGather from all Context workers in DWDP group
        all_data = self.dwdp_group.allgather(local_data)
        
        # Open handles from peer workers
        for peer_data in all_data:
            peer_rank = peer_data['rank']
            self.peer_expert_ranges[peer_rank] = (peer_data['expert_start_id'], peer_data['expert_end_id'])
            
            # Save samples from all ranks (including self) for verification
            for layer_idx, ipc_collector in enumerate(peer_data['ipc_collectors']):
                peer_samples = ipc_collector.get('samples', {})
                for param_name, samples in peer_samples.items():
                    key = (layer_idx, param_name)
                    if key not in self.all_samples:
                        self.all_samples[key] = {}
                    self.all_samples[key][peer_rank] = samples
            
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
            num_prefetch_experts=self.num_prefetch_experts,
            num_layers=len(self.ipc_collectors),
            param_shapes=self.ipc_collectors[0].param_shapes,
            param_dtypes=self.ipc_collectors[0].param_dtypes,
        )
        self.prefetch_buffer.initialize_compute_events()
        
    def prefetch_first_layers(self):
        """Prefetch the first num_buffers layers as warmup."""
        assert self.prefetch_buffer is not None, "Prefetch buffer is not initialized"
        for layer_idx in range(3, 3 + self.prefetch_buffer.num_buffers):
            self.prefetch_layer(layer_idx)
            self.prefetch_buffer.record_prefetch_event(layer_idx)
            logger.info(f"[DWDP] Warmup prefetch completed for layer {layer_idx}")

    def wait_prefetch_and_get_buffer(self, layer_idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Wait for prefetch to complete and return the buffer for this layer."""
        assert self.prefetch_buffer is not None, "Prefetch buffer is not initialized"
        self.prefetch_buffer.wait_prefetch_event(layer_idx)
        buffer_idx = layer_idx % self.prefetch_buffer.num_buffers
        logger.info(f"[DWDP] Wait prefetch and get buffer for layer {layer_idx} buffer_idx: {buffer_idx}")
        return self.prefetch_buffer.buffers[buffer_idx]

    def record_compute_and_prefetch_next(self, layer_idx: int):
        """Record compute completion and trigger prefetch for layer_idx + num_buffers."""
        assert self.prefetch_buffer is not None, "Prefetch buffer is not initialized"
        # Record compute event for current layer
        self.prefetch_buffer.record_compute_event(layer_idx)

        next_layer_idx = layer_idx + self.prefetch_buffer.num_buffers
        if next_layer_idx >= self.prefetch_buffer.max_layer_idx:
            return
        # prefetch_layer handles stream internally: local copy on default stream, peer copy on prefetch stream
        self.prefetch_layer(next_layer_idx, wait_compute_layer_idx=layer_idx)
        self.prefetch_buffer.record_prefetch_event(next_layer_idx)
        logger.info(f"[DWDP] Record compute and prefetch next for layer {layer_idx} next_layer_idx: {next_layer_idx}")

    def verify_prefetch_buffer(self, layer_idx: int):
        """
        Verify prefetch buffer contains correct data by comparing with original samples.
        
        The ground truth is built by concatenating samples from all ranks in order.
        Then we sample the prefetch buffer at the same positions and compare.
        """
        buffer_idx = layer_idx % self.prefetch_buffer.num_buffers
        collector = self.ipc_collectors[layer_idx]
        
        for param_name in collector.param_shapes.keys():
            key = (layer_idx, param_name)
            # Build ground truth: concatenate samples from all ranks in order
            # Each rank samples at [0, 8, 16, ..., 56] within their local experts
            ground_truth = []
            for rank in range(self.dwdp_size):
                ground_truth.extend(self.all_samples[key][rank])
            
            # Sample prefetch buffer at the same positions
            # Global positions: rank0's [0,8,16,...] + rank1's [0,8,16,...] + ...
            # = [0, 8, 16, ..., 56, experts_per_worker+0, experts_per_worker+8, ...]
            buffer_tensor = self.prefetch_buffer.buffers[buffer_idx][param_name]
            
            buffer_samples = []
            for rank in range(self.dwdp_size):
                rank_start = self.peer_expert_ranges[rank][0]
                sample_positions = list(range(0, min(64, self.experts_per_worker), 8))
                for local_expert_idx in sample_positions:
                    global_expert_idx = rank_start + local_expert_idx
                    sample_value = buffer_tensor[global_expert_idx].flatten()[0].item()
                    buffer_samples.append(sample_value)
            
            # Compare
            if len(ground_truth) != len(buffer_samples):
                logger.error(f"[DWDP Verify] ❌ Layer {layer_idx} param {param_name}: "
                           f"length mismatch ground_truth={len(ground_truth)} buffer={len(buffer_samples)}")
                continue
            
            match = True
            for i, (gt, buf) in enumerate(zip(ground_truth, buffer_samples)):
                if gt != buf:
                    match = False
                    logger.error(f"[DWDP Verify] ❌ Layer {layer_idx} param {param_name} idx {i}: "
                               f"expected {gt}, got {buf}")
            
            if match:
                logger.info(f"[DWDP Verify] ✅ Layer {layer_idx} param {param_name}: "
                          f"all {len(ground_truth)} samples match! first_4={ground_truth[:4]}")
            

    def _get_prefetch_range_from_peer(self, peer_rank: int) -> Tuple[int, int, int, int]:
        """
        Calculate what experts to fetch from a peer and where to put them.
        
        Returns:
            (src_offset, dst_global_id)
            
        Example: 256 experts, rank0: [0, 200), rank1: [56, 256)
        - rank0 needs [200, 256) from rank1:
          src_offset = 200 - 56 = 144, dst_global_id = 200, count = 56
        - rank1 needs [0, 56) from rank0:
          src_offset = 0 - 0 = 0, dst_global_id = 0, count = 56
        """
        peer_start, peer_end = self.peer_expert_ranges[peer_rank]
        
        # What I need = global - what I have
        # From peer = what I need ∩ what peer has
        if self.rank < peer_rank:
            # I'm earlier rank, need experts after my end
            prefetch_end = peer_end
            prefetch_start = prefetch_end - self.num_prefetch_experts
        else:
            # I'm later rank, need experts before my start
            prefetch_start = peer_start
            prefetch_end = prefetch_start + self.num_prefetch_experts
        
        src_offset = prefetch_start - peer_start
        dst_global_id = prefetch_start
        
        return src_offset, dst_global_id

    @nvtx_range("dwdp_prefetch_layer")
    def prefetch_layer(self, layer_idx: int, wait_compute_layer_idx: Optional[int] = None):
        """
        Prefetch layer data from local and peer ranks.
        
        Args:
            layer_idx: The layer to prefetch
            wait_compute_layer_idx: If provided, wait for this layer's compute to complete
                                    before overwriting buffer (used when prefetching next layer)
        
        Local copy runs on default stream, peer copy runs on prefetch stream.
        """
        param_names = self.ipc_collectors[layer_idx-3].param_shapes.keys()
        collector = self.ipc_collectors[layer_idx-3]
        buffer_idx = layer_idx % self.prefetch_buffer.num_buffers

        # Step 1: Local copy on default stream
        # No need to wait for compute_event here because local copy and compute
        # are on the same stream (default stream), so they are naturally serialized
        for param_name in param_names:
            param_shape = collector.param_shapes[param_name]
            param_dtype = collector.param_dtypes[param_name]
            expert_size = param_shape.numel() * param_dtype.itemsize
            
            # src_ptr is local param data
            src_ptr = collector.local_ptrs[param_name]
            
            # dst_ptr at local expert range in prefetch buffer
            dst_tensor = self.prefetch_buffer.buffers[buffer_idx][param_name]
            dst_ptr = dst_tensor.data_ptr() + self.start_expert_id * expert_size
            
            data_size = self.experts_per_worker * expert_size
            
            err, = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                data_size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                torch.cuda.current_stream().cuda_stream,
            )
            check_cuda_error(err, f"prefetch layer {layer_idx} local copy {param_name}")
        logger.info(f"prefetch layer {layer_idx} local copy to [{self.start_expert_id}, {self.end_expert_id})")

        # Step 2: Peer copy on prefetch stream
        with torch.cuda.stream(self.prefetch_buffer.prefetch_stream):
            # Also wait on prefetch stream (both streams need to wait for compute)
            if wait_compute_layer_idx is not None:
                self.prefetch_buffer.wait_compute_event(wait_compute_layer_idx)
            
            for peer_rank in range(self.dwdp_size):
                if peer_rank == self.rank:
                    continue  # Skip local rank, already handled above
                
                src_expert_offset, dst_global_id = self._get_prefetch_range_from_peer(peer_rank)
                logger.info(f"prefetch layer {layer_idx} rank {peer_rank} src_expert_offset: {src_expert_offset} dst_global_id: {dst_global_id}")
                
                for param_name in param_names:
                    param_shape = collector.param_shapes[param_name]
                    param_dtype = collector.param_dtypes[param_name]
                    expert_size = param_shape.numel() * param_dtype.itemsize
                    
                    # src_ptr points to peer's tensor start, add offset for specific experts
                    base_ptr = collector.get_peer_ptr(peer_rank, param_name)
                    src_ptr = base_ptr + src_expert_offset * expert_size
                    
                    # dst_ptr in prefetch buffer using global expert id
                    dst_tensor = self.prefetch_buffer.buffers[buffer_idx][param_name]
                    dst_ptr = dst_tensor.data_ptr() + dst_global_id * expert_size
                    
                    data_size = self.num_prefetch_experts * expert_size

                    err, = cudart.cudaMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        data_size,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        self.prefetch_buffer.prefetch_stream.cuda_stream,
                    )
                    check_cuda_error(err, f"prefetch layer {layer_idx} rank {peer_rank} {param_name}")