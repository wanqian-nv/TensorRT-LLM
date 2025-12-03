import pickle
import hashlib
from typing import Any, List, NewType

from .llm_request import LlmRequest
from tensorrt_llm.logger import logger

NONE_HASH = b""

def hash_func(token_ids: Any, parent_block_hash: Any) -> str:
    def sha256(input: Any) -> bytes:
        """Hash any picklable Python object using SHA-256.

        The input is serialized using pickle before hashing, which allows
        arbitrary Python objects to be used. Note that this function does
        not use a hash seed—if you need one, prepend it explicitly to the input.

        Args:
            input: Any picklable Python object.

        Returns:
            Bytes representing the SHA-256 hash of the serialized input.
        """
        input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(input_bytes).digest()
    
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    token_ids = tuple(token_ids)
    
    return sha256((parent_block_hash, token_ids))

class FreeBlocksNode:
    def __init__(self, block_hash: str):
        self.block_hash = block_hash
        self.next = None
        self.prev = None

class FreeBlocksList:
    def __init__(self):
        self.head = FreeBlocksNode('HEAD')
        self.tail = FreeBlocksNode('TAIL')
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hash_to_node = {}

    def add(self, block_hash):
        node = FreeBlocksNode(block_hash)
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
        self.hash_to_node[block_hash] = node

    def remove(self, block_hash):
        node = self.hash_to_node[block_hash]
        node.prev.next = node.next
        node.next.prev = node.prev
        del self.hash_to_node[block_hash]

    def pop_left(self) -> str:
        if self.head.next == self.tail:
            return None
        node = self.head.next
        self.remove(node.block_hash)
        return node.block_hash


class LazyPrefixTree:
    def __init__(self, max_num_blocks: int, block_size: int = 32):

        # Block hash -> Reference count
        self.hash_table = {}
        self.free_blocks_list = FreeBlocksList()
        self.block_size = block_size
        self.num_blocks = 0
        self.max_num_blocks = max_num_blocks

    def match(self, request: LlmRequest) -> int:
        """
        Match the request with the lazy prefix tree and return the number of reused tokens
        """
        reused_tokens = 0
        for hash in request.block_hash:
            ref_cnt = self.hash_table.get(hash, None)
            if ref_cnt is None:
                return reused_tokens
            reused_tokens += self.block_size
        return reused_tokens

    def insert_request(self, request: LlmRequest):
        """
        Insert the request into the lazy prefix tree
        """
        for hash in request.block_hash:
            ref_cnt = self.hash_table.get(hash, None)
            if ref_cnt is None:
                self.hash_table[hash] = 1
                self.num_blocks += 1
            elif ref_cnt == 0:
                self.free_blocks_list.remove(hash)
                self.hash_table[hash] += 1
            else:
                self.hash_table[hash] += 1

    def free_request(self, request: LlmRequest):
        """
        Remove the request from the lazy prefix tree
        """
        for hash in request.block_hash:
            ref_cnt = self.hash_table[hash]
            ref_cnt -= 1
            assert ref_cnt >= 0
            if ref_cnt == 0:
                self.free_blocks_list.add(hash)
            self.hash_table[hash] = ref_cnt

    def lazy_evict(self):
        """
        Evict the least recently used blocks from the free blocks list
        """
        while self.num_blocks >= self.max_num_blocks:
            block_hash = self.free_blocks_list.pop_left()
            assert self.hash_table[block_hash] == 0
            del self.hash_table[block_hash]
            self.num_blocks -= 1

# Lazy Prefix Tree
# A request comes
# 1. Calculate the block hashes of the request, and determine the reused/missed tokens
# 2. When cache hit, update the hash table and ref_cnt-1
# 3. When cache miss, insert the block into the hash table and ref_cnt=1
# 4. When the request is finished, ref_cnt-1 and add the block into the free_blocks
# 5. After a certain number of iterations (based on the accumulated allocated blocks), remove the free_blocks from the hash table based on the LRU policy