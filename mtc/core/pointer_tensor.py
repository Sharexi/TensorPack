"""
PointerTensor: Hierarchical Recursive Composition and Computational DAG for MTC-8
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from .types import OpCode, PrecisionMode
from .arena import MemoryArena
from .microtensor import MicroTensor, BlockDescriptor


class PointerTensor:
    """
    Structural tensor layer supporting:
    1. Leaf data references into memory arena.
    2. Hierarchical grid composition (e.g. 2x2 grid of sub-blocks).
    3. Computational DAG nodes (MATMUL, ADD, SCALE) referencing operand pointers.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        op_code: OpCode = OpCode.DATA,
        microtensor: Optional[MicroTensor] = None,
        children: Optional[List[List["PointerTensor"]]] = None,
        operands: Optional[List["PointerTensor"]] = None,
        scale: float = 1.0
    ):
        self.shape = shape
        self.op_code = op_code
        self.microtensor = microtensor
        self.children = children      # 2D list for grid composition
        self.operands = operands      # 1D list for computational ops
        self.scale = scale

    @property
    def is_leaf(self) -> bool:
        return self.microtensor is not None

    @property
    def is_composite(self) -> bool:
        return self.children is not None

    @property
    def is_op(self) -> bool:
        return self.op_code != OpCode.DATA

    @classmethod
    def from_microtensor(cls, mt: MicroTensor) -> "PointerTensor":
        """Wrap an elementary micro-tensor as a leaf pointer node."""
        return cls(
            shape=mt.shape,
            op_code=OpCode.DATA,
            microtensor=mt,
            scale=1.0
        )

    @classmethod
    def from_grid(cls, grid: List[List["PointerTensor"]]) -> "PointerTensor":
        """
        Recursively compose a higher-level tensor from a 2D grid of PointerTensors.
        """
        grid_rows = len(grid)
        grid_cols = len(grid[0])
        sub_h, sub_w = grid[0][0].shape

        total_h = sum(grid[r][0].shape[0] for r in range(grid_rows))
        total_w = sum(grid[0][c].shape[1] for c in range(grid_cols))

        return cls(
            shape=(total_h, total_w),
            op_code=OpCode.DATA,
            children=grid
        )

    @classmethod
    def from_matrix_recursive(
        cls,
        matrix: np.ndarray,
        leaf_shape: Tuple[int, int],
        precision: PrecisionMode,
        arena: MemoryArena,
        block_scale: Optional[float] = None
    ) -> "PointerTensor":
        """
        Recursively construct a quadtree PointerTensor from an (H, W) matrix.
        Base case: leaf_shape reached -> creates MicroTensor.
        """
        H, W = matrix.shape
        if (H, W) == leaf_shape:
            mt = MicroTensor.from_array(matrix, precision, arena, scale=block_scale)
            return cls.from_microtensor(mt)

        # Split into 2x2 quadrants
        mid_h = H // 2
        mid_w = W // 2

        q00 = cls.from_matrix_recursive(matrix[:mid_h, :mid_w], leaf_shape, precision, arena, block_scale)
        q01 = cls.from_matrix_recursive(matrix[:mid_h, mid_w:], leaf_shape, precision, arena, block_scale)
        q10 = cls.from_matrix_recursive(matrix[mid_h:, :mid_w], leaf_shape, precision, arena, block_scale)
        q11 = cls.from_matrix_recursive(matrix[mid_h:, mid_w:], leaf_shape, precision, arena, block_scale)

        return cls.from_grid([[q00, q01], [q10, q11]])

    def to_array(self) -> np.ndarray:
        """
        Resolve hierarchy into a full floating-point NumPy matrix.
        Evaluates DAG operations if node is an OP_REF.
        """
        if self.is_op:
            return self.evaluate()

        if self.is_leaf:
            return self.microtensor.unpack_to_array() * self.scale

        if self.is_composite:
            rows = []
            for row_children in self.children:
                row_blocks = [c.to_array() for c in row_children]
                rows.append(np.hstack(row_blocks))
            return np.vstack(rows) * self.scale

        raise RuntimeError("Invalid PointerTensor state")

    def __matmul__(self, other: "PointerTensor") -> "PointerTensor":
        """Lazy matrix multiplication DAG node."""
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        out_shape = (self.shape[0], other.shape[1])
        return PointerTensor(
            shape=out_shape,
            op_code=OpCode.MATMUL,
            operands=[self, other]
        )

    def __add__(self, other: "PointerTensor") -> "PointerTensor":
        """Lazy addition DAG node."""
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return PointerTensor(
            shape=self.shape,
            op_code=OpCode.ADD,
            operands=[self, other]
        )

    def __mul__(self, scalar: float) -> "PointerTensor":
        """Lazy scalar scale DAG node."""
        return PointerTensor(
            shape=self.shape,
            op_code=OpCode.SCALE,
            operands=[self],
            scale=float(scalar)
        )

    def transpose(self) -> "PointerTensor":
        """
        Zero-copy transposition of pointer structure.
        Swaps child pointers or registers transpose opcode.
        """
        if self.is_leaf:
            unpacked = self.microtensor.unpack_to_array().T
            mt_t = MicroTensor.from_array(
                unpacked,
                self.microtensor.precision,
                self.microtensor.arena,
                scale=self.microtensor.scale
            )
            return PointerTensor.from_microtensor(mt_t)

        if self.is_composite:
            num_r = len(self.children)
            num_c = len(self.children[0])
            t_children = [[self.children[r][c].transpose() for r in range(num_r)] for c in range(num_c)]
            return PointerTensor(
                shape=(self.shape[1], self.shape[0]),
                op_code=OpCode.DATA,
                children=t_children,
                scale=self.scale
            )

        return PointerTensor(
            shape=(self.shape[1], self.shape[0]),
            op_code=OpCode.TRANSPOSE,
            operands=[self],
            scale=self.scale
        )

    @property
    def T(self) -> "PointerTensor":
        return self.transpose()

    def evaluate(self) -> np.ndarray:
        """
        Evaluate computational DAG node into concrete NumPy array.
        """
        if not self.is_op:
            return self.to_array()

        if self.op_code == OpCode.MATMUL:
            A = self.operands[0].evaluate() if self.operands[0].is_op else self.operands[0].to_array()
            B = self.operands[1].evaluate() if self.operands[1].is_op else self.operands[1].to_array()
            return (A @ B) * self.scale

        elif self.op_code == OpCode.ADD:
            A = self.operands[0].evaluate() if self.operands[0].is_op else self.operands[0].to_array()
            B = self.operands[1].evaluate() if self.operands[1].is_op else self.operands[1].to_array()
            return (A + B) * self.scale

        elif self.op_code == OpCode.SCALE:
            A = self.operands[0].evaluate() if self.operands[0].is_op else self.operands[0].to_array()
            return A * self.scale

        elif self.op_code == OpCode.TRANSPOSE:
            A = self.operands[0].evaluate() if self.operands[0].is_op else self.operands[0].to_array()
            return A.T * self.scale

        raise NotImplementedError(f"OpCode {self.op_code} evaluation not implemented")

    def count_nodes(self) -> Tuple[int, int]:
        """Return (num_data_nodes, num_op_nodes) in tree."""
        if self.is_leaf:
            return (1, 0)
        data_count = 0
        op_count = 1 if self.is_op else 0
        if self.children:
            for row in self.children:
                for child in row:
                    d, o = child.count_nodes()
                    data_count += d
                    op_count += o
        if self.operands:
            for op in self.operands:
                d, o = op.count_nodes()
                data_count += d
                op_count += o
        return (data_count, op_count)
