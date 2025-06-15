"""Module for storing unique 3D voxel observations as sparse tensors in a buffer.

This is designed for RL environments with voxel observations where the same
voxel tensors are frequently repeated. Instead of storing duplicates, we store
unique tensors once and reference them by ID.

The buffer maintains a mapping of active IDs and can be periodically cleaned
up to remove unused voxel tensors.

Example:
    buffer = SparseVoxelBuffer()
    
    # Add voxels
    voxel1 = torch.zeros(32, 32, 32, dtype=torch.float32)
    vid1 = buffer.add(voxel1)
    
    # Get voxels by ID
    retrieved = buffer.get(vid1)
    
    # Periodically clean up unused voxels
    active_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    buffer.cleanup(active_ids)
"""

import torch
from typing import List, Optional, Tuple, Dict


class SparseVoxelBuffer:
    """Buffer that stores unique 3D voxel tensors as sparse tensors.
    
    Voxels are stored in COO sparse format and referenced by integer IDs.
    The buffer maintains a free list of available IDs for reuse.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the sparse voxel buffer.
        
        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.device = device
        # Maps voxel ID to sparse tensor data
        self._voxels: Dict[int, torch.Tensor] = {}
        # Maps sparse tensor coordinates to voxel ID (for deduplication)
        self._coords_to_id: Dict[Tuple[torch.Tensor, torch.Tensor], int] = {}
        # Set of currently used voxel IDs
        self._active_ids: torch.Tensor = torch.tensor([], dtype=torch.long, device=device)
        # List of free IDs that can be reused
        self._free_ids: List[int] = []
        # Next available ID (when no free IDs are available)
        self._next_id = 0

    def add(self, voxel: torch.Tensor) -> int:
        """Add a 3D voxel tensor to the buffer if not already present.
        
        Args:
            voxel: 3D tensor of shape (D, H, W) to add
            
        Returns:
            int: The ID corresponding to the voxel in the buffer
        """
        # Convert to sparse COO format
        sparse_voxel = voxel.to_sparse()
        coords = sparse_voxel.indices()
        values = sparse_voxel.values()
        
        # Check if we've seen this voxel before
        key = (coords, values)
        if key in self._coords_to_id:
            return self._coords_to_id[key]
        
        # Get a new ID (either from free list or next available)
        if self._free_ids:
            voxel_id = self._free_ids.pop()
        else:
            voxel_id = self._next_id
            self._next_id += 1
        
        # Store the voxel
        self._voxels[voxel_id] = sparse_voxel.to(self.device)
        self._coords_to_id[key] = voxel_id
        
        return voxel_id

    def get(self, voxel_id: int, shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """Retrieve a voxel tensor by its ID.
        
        Args:
            voxel_id: The ID of the voxel to retrieve
            shape: Optional shape to reshape the dense tensor to
            
        Returns:
            torch.Tensor: The stored sparse voxel tensor
        """
        if voxel_id not in self._voxels:
            raise ValueError(f"Voxel ID {voxel_id} not found in buffer")
            
        sparse_tensor = self._voxels[voxel_id]
        if shape is not None:
            return sparse_tensor.to_dense().view(shape)
        return sparse_tensor

    def cleanup(self, active_ids: torch.Tensor) -> None:
        """Remove voxels that are no longer in use.
        
        Args:
            active_ids: Tensor of voxel IDs that are *currently* in use
        """
        if not isinstance(active_ids, torch.Tensor):
            active_ids = torch.tensor(active_ids, dtype=torch.long, device=self.device)
            
        active_set = set(active_ids.tolist())
        
        # Find IDs to remove
        to_remove = []
        for vid in list(self._voxels.keys()):
            if vid not in active_set:
                to_remove.append(vid)
        
        # Remove unused voxels and add their IDs to free list
        for vid in to_remove:
            # Remove from both mappings
            sparse_tensor = self._voxels.pop(vid)
            coords = sparse_tensor.indices()
            values = sparse_tensor.values()
            self._coords_to_id.pop((coords, values), None)
            
            # Add to free list
            self._free_ids.append(vid)
    
    def get_active_ids(self) -> torch.Tensor:
        """Get a tensor of all active voxel IDs."""
        return torch.tensor(list(self._voxels.keys()), dtype=torch.long, device=self.device)
    
    def __len__(self) -> int:
        """Get the number of unique voxels stored."""
        return len(self._voxels)
    
    def clear(self) -> None:
        """Clear all stored voxels from the buffer."""
        self._voxels.clear()
        self._coords_to_id.clear()
        self._free_ids = []
        self._next_id = 0
