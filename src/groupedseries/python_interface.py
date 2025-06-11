#!/usr/bin/env python3
"""
Python interface to the Nim GroupedSeries implementation.

This module demonstrates how to use the Nim-based GroupedSeries
from Python, bridging the gap between the native Nim implementation
and Python data science workflows.
"""

import numpy as np
from typing import List, Tuple, Optional, Any

# Import the compiled Nim module (this would be the result of compiling groupedseries.nim)
# In practice, you'd compile with: nim c --app:lib --out:groupedseries.so groupedseries.nim
try:
    import groupedseries_nim  # This would be the compiled Nim module

except ImportError:
    # Fallback for when the Nim module isn't available
    print("Warning: Nim GroupedSeries module not available. This is a demonstration only.")
    groupedseries_nim = None

class GroupedSeries:
    """
    Python wrapper for the Nim GroupedSeries implementation.

    This class provides a Pythonic interface to the high-performance
    Nim GroupedSeries while maintaining compatibility with numpy arrays.
    """

    def __init__(
        self,
        epochs: np.ndarray,
        groups: List[List[str]],
        values: np.ndarray,
        is_interpolated: np.ndarray,
        dense_epochs: Optional[np.ndarray] = None,
        keep_interpolated: bool = False,
        interval: Optional[int] = None
    ):
        """
        Initialize a GroupedSeries.

        Args:
            epochs: 1D numpy array of int32 timestamps
            groups: List of group tag lists, where each tag is "key:value"
            values: 2D numpy array of float64, shape (groups, epochs)
            is_interpolated: 1D numpy array of bool, length matches epochs
            dense_epochs: Optional dense epochs array
            keep_interpolated: Whether to keep interpolated values
            interval: Optional interval value
        """
        self.epochs = np.asarray(epochs, dtype=np.int32)
        self.groups = groups
        self.values = np.asarray(values, dtype=np.float64)
        self.is_interpolated = np.asarray(is_interpolated, dtype=bool)
        self.dense_epochs = dense_epochs
        self.keep_interpolated = keep_interpolated
        self.interval = interval

        # Validate dimensions
        if len(self.values.shape) == 2:
            assert self.values.shape[0] == len(self.groups), \
                f"Values shape[0] ({self.values.shape[0]}) must match groups length ({len(self.groups)})"
            assert self.values.shape[1] == len(self.epochs), \
                f"Values shape[1] ({self.values.shape[1]}) must match epochs length ({len(self.epochs)})"

        assert len(self.is_interpolated) == len(self.epochs), \
            f"is_interpolated length ({len(self.is_interpolated)}) must match epochs length ({len(self.epochs)})"

        # Create the underlying Nim object if available
        self._nim_gs = None
        if groupedseries_nim:
            try:
                self._nim_gs = groupedseries_nim.GroupedSeries(
                    self.epochs, self.groups, self.values, self.is_interpolated
                )
            except Exception as e:
                print(f"Warning: Could not create Nim GroupedSeries: {e}")

    @classmethod
    def empty(
        cls,
        epochs: Optional[np.ndarray] = None,
        groups: Optional[List[List[str]]] = None,
        dense_epochs: Optional[np.ndarray] = None,
        keep_interpolated: bool = False,
        interval: Optional[int] = None
    ) -> 'GroupedSeries':
        """Create an empty GroupedSeries."""
        if epochs is None:
            epochs = np.array([], dtype=np.int32)
        if groups is None:
            groups = []

        is_interpolated = np.zeros(len(epochs), dtype=bool)
        values = np.full((len(groups), len(epochs)), np.nan, dtype=np.float64)

        return cls(epochs, groups, values, is_interpolated, dense_epochs, keep_interpolated, interval)

    @classmethod
    def from_dict(cls, data: dict) -> 'GroupedSeries':
        """
        Create GroupedSeries from a dictionary representation.

        Args:
            data: Dictionary with keys 'epochs', 'groups', 'values', 'is_interpolated'
        """
        return cls(
            data['epochs'],
            data['groups'],
            data['values'],
            data['is_interpolated'],
            data.get('dense_epochs'),
            data.get('keep_interpolated', False),
            data.get('interval')
        )

    def num_groups(self) -> int:
        """Return number of groups."""
        if self._nim_gs and groupedseries_nim:
            return groupedseries_nim.num_groups(self._nim_gs)
        return len(self.groups)

    def num_epochs(self) -> int:
        """Return number of epochs."""
        if self._nim_gs and groupedseries_nim:
            return groupedseries_nim.num_epochs(self._nim_gs)
        return len(self.epochs)

    def is_empty(self) -> bool:
        """Check if GroupedSeries is empty."""
        if self._nim_gs and groupedseries_nim:
            return groupedseries_nim.is_empty(self._nim_gs)
        return self.values.size == 0

    def get_values(self) -> np.ndarray:
        """Get the values array."""
        if self._nim_gs and groupedseries_nim:
            return groupedseries_nim.get_values(self._nim_gs)
        return self.values

    def get_epochs(self) -> np.ndarray:
        """Get the epochs array."""
        if self._nim_gs and groupedseries_nim:
            return groupedseries_nim.get_epochs(self._nim_gs)
        return self.epochs

    def get_groups(self) -> List[List[str]]:
        """Get the groups list."""
        if self._nim_gs and groupedseries_nim:
            return groupedseries_nim.get_groups(self._nim_gs)
        return self.groups

    def view(self, group_idx: int) -> np.ndarray:
        """Get values for a single group."""
        assert 0 <= group_idx < self.num_groups(), f"Group index {group_idx} out of range"
        return self.values[group_idx, :]

    def view_groups(self, target_groups: List[List[str]]) -> np.ndarray:
        """Get values for specific groups in the requested order."""
        group_indices = []
        for target_group in target_groups:
            try:
                idx = self.groups.index(target_group)
                group_indices.append(idx)
            except ValueError:
                raise ValueError(f"Group {target_group} not found")

        return self.values[group_indices, :]

    def copy(self) -> 'GroupedSeries':
        """Create a deep copy."""
        return GroupedSeries(
            self.epochs.copy(),
            [group.copy() for group in self.groups],
            self.values.copy(),
            self.is_interpolated.copy(),
            self.dense_epochs.copy() if self.dense_epochs is not None else None,
            self.keep_interpolated,
            self.interval
        )

    def remove_interpolated(self) -> None:
        """Remove interpolated data points in-place."""
        if self.keep_interpolated:
            return

        # Find non-interpolated indices
        kept_mask = ~self.is_interpolated

        if not np.any(kept_mask):
            # Remove all data
            self.values = np.array([], dtype=np.float64).reshape(0, 0)
            self.epochs = np.array([], dtype=np.int32)
            self.is_interpolated = np.array([], dtype=bool)
            self.groups = []
            return

        # Filter arrays
        self.epochs = self.epochs[kept_mask]
        self.is_interpolated = self.is_interpolated[kept_mask]
        if len(self.values.shape) == 2:
            self.values = self.values[:, kept_mask]

    def add_group(self, group_name: List[str], data: np.ndarray) -> None:
        """Add a new group with data."""
        assert len(data) == self.num_epochs(), \
            f"Data length ({len(data)}) must match epochs length ({self.num_epochs()})"

        self.groups.append(group_name)

        # Expand values array
        if len(self.values.shape) == 1 or self.values.size == 0:
            self.values = data.reshape(1, -1)
        else:
            self.values = np.vstack([self.values, data.reshape(1, -1)])

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        # Convert epochs to datetime index
        index = pd.to_datetime(self.epochs, unit='s')

        # Create column names from groups
        columns = []
        for group in self.groups:
            # Join group tags with '|' separator
            col_name = '|'.join(group)
            columns.append(col_name)

        return pd.DataFrame(self.values.T, index=index, columns=columns)

    def __repr__(self) -> str:
        """String representation."""
        return (f"<NimGroupedSeries(n_epochs={self.num_epochs()}, "
                f"n_groups={self.num_groups()}, values_shape={self.values.shape}, "
                f"interval={self.interval})>")

    def __str__(self) -> str:
        """String representation for display."""
        try:
            return self.to_dataframe().to_string()
        except ImportError:
            return self.__repr__()


def create_sample_data() -> GroupedSeries:
    """Create sample GroupedSeries data for testing."""
    # Sample timestamps (4 time points)
    epochs = np.array([1640995200, 1640995260, 1640995320, 1640995380], dtype=np.int32)

    # Sample groups - each group is a list of tag strings (equivalent to Python tuples)
    # This mirrors the structure that will be passed to the Nim implementation
    groups = [
        ["host:server1", "metric:cpu_usage", "datacenter:us-east"],
        ["host:server2", "metric:cpu_usage", "datacenter:us-east"],
        ["host:server1", "metric:memory_usage", "datacenter:us-east"],
        ["host:server3", "metric:cpu_usage", "datacenter:us-west"]
    ]

    # Sample values (4 groups x 4 epochs)
    values = np.array([
        [50.0, 55.0, 60.0, 58.0],  # Server1 CPU
        [45.0, 50.0, 48.0, 52.0],  # Server2 CPU
        [75.0, 78.0, 80.0, 77.0],  # Server1 Memory
        [42.0, 47.0, 44.0, 49.0]   # Server3 CPU
    ], dtype=np.float64)

    # No interpolated values
    is_interpolated = np.array([False, False, False, False], dtype=bool)

    return GroupedSeries(epochs, groups, values, is_interpolated)


def main():
    """Demonstrate the GroupedSeries functionality."""
    print("=== GroupedSeries Python Interface Demo ===")

    # Create sample data
    gs = create_sample_data()
    print(f"Created GroupedSeries: {gs}")

    # Test basic methods
    print(f"\nNumber of groups: {gs.num_groups()}")
    print(f"Number of epochs: {gs.num_epochs()}")
    print(f"Is empty: {gs.is_empty()}")

    # Test data access
    print("\nGroups:")
    for i, group in enumerate(gs.get_groups()):
        print(f"  Group {i}: {group}")

    print("\nValues for first group:", gs.view(0))

    # Test copy
    gs_copy = gs.copy()
    print(f"\nCopied GroupedSeries: {gs_copy}")

    # Test adding a group
    new_data = np.array([30.0, 32.0, 35.0, 33.0], dtype=np.float64)
    gs_copy.add_group(["host:server2", "metric:memory_usage"], new_data)
    print(f"After adding group: {gs_copy}")

    # Test DataFrame conversion (if pandas available)
    try:
        df = gs.to_dataframe()
        print("\nAs DataFrame:")
        print(df)
    except ImportError:
        print("\nDataFrame conversion requires pandas")

    # Test empty GroupedSeries
    empty_gs = NimGroupedSeries.empty()
    print(f"\nEmpty GroupedSeries: {empty_gs}")


if __name__ == "__main__":
    main()