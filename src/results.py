from dataclasses import dataclass
from typing import List, Optional
import torch
import pandas as pd


@dataclass
class VARXResult:
    """Results container for VAR and OR-VARX models.

    Attributes:
        forecasts: Selected forecasts using optimal lag, shape (n_assets, n_output_days)
                  where n_output_days = n_total_test_days - validation_days
        forecasts_all: All forecasts for each lag, shape (n_assets, n_output_days, p_max)
        p_optimal: Optimal lag order for each day, shape (n_output_days,)
                  Selected via rolling validation window for each output day
        p_max: Maximum lag order considered
        coefficients: Learned coefficients, shape (n_output_days, p_max, n_assets, n_assets)
                     coefficients[d, k, i, j] = effect of asset j at lag k+1 on asset i for day d
        asset_names: List of asset names
        confounder_names: List of confounder names (empty for VAR, non-empty for OR-VARX)
        dates: List of date strings for each output day (length n_output_days)
    """
    forecasts: torch.Tensor
    forecasts_all: torch.Tensor
    p_optimal: torch.Tensor
    p_max: int
    coefficients: torch.Tensor
    asset_names: List[str]
    confounder_names: List[str]
    dates: List[str]

    @property
    def method(self) -> str:
        """Returns model type: 'VAR' or 'OR-VARX'."""
        return "OR-VARX" if len(self.confounder_names) > 0 else "VAR"

    @property
    def is_orthogonalized(self) -> bool:
        """Returns True if model uses orthogonalization."""
        return len(self.confounder_names) > 0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecasts to DataFrame with dates as index and assets as columns.

        Returns:
            DataFrame with shape (n_days, n_assets)
        """
        return pd.DataFrame(
            self.forecasts.cpu().numpy().T,  # Transpose to (n_days, n_assets)
            index=self.dates,
            columns=self.asset_names
        )

    def get_leadlag_matrix(self, day_idx: int, lag: int) -> pd.DataFrame:
        """Extract lead-lag coefficient matrix for a specific day and lag.

        Args:
            day_idx: Index of the forecast day
            lag: Lag order (1-indexed, must be in range [1, p_max])

        Returns:
            DataFrame with shape (n_assets, n_assets) where entry [i, j] represents
            the effect of asset j at the given lag on asset i
        """
        if not 1 <= lag <= self.p_max:
            raise ValueError(f"Lag must be in range [1, {self.p_max}], got {lag}")
        if not 0 <= day_idx < len(self.dates):
            raise ValueError(f"Day index must be in range [0, {len(self.dates)}), got {day_idx}")

        # coefficients shape: (n_days, p_max, n_assets, n_assets)
        # lag is 1-indexed, so we use lag-1 for 0-indexed access
        coef_matrix = self.coefficients[day_idx, lag - 1].cpu().numpy()

        return pd.DataFrame(
            coef_matrix,
            index=self.asset_names,
            columns=self.asset_names
        )

    def save(self, path: str) -> None:
        """Save results to disk using torch.save.

        Args:
            path: File path to save to (typically with .pt or .pth extension)
        """
        save_dict = {
            'forecasts': self.forecasts,
            'forecasts_all': self.forecasts_all,
            'p_optimal': self.p_optimal,
            'p_max': self.p_max,
            'coefficients': self.coefficients,
            'asset_names': self.asset_names,
            'confounder_names': self.confounder_names,
            'dates': self.dates,
            'result_type': 'VARXResult'
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None) -> 'VARXResult':
        """Load results from disk.

        Args:
            path: File path to load from
            device: Device to load tensors to (default: None, keeps original device)

        Returns:
            VARXResult instance
        """
        if device is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=device)

        if checkpoint.get('result_type') != 'VARXResult':
            raise ValueError(f"File does not contain VARXResult data")

        return VARXResult(
            forecasts=checkpoint['forecasts'],
            forecasts_all=checkpoint['forecasts_all'],
            p_optimal=checkpoint['p_optimal'],
            p_max=checkpoint['p_max'],
            coefficients=checkpoint['coefficients'],
            asset_names=checkpoint['asset_names'],
            confounder_names=checkpoint['confounder_names'],
            dates=checkpoint['dates']
        )


@dataclass
class ORACLEVARXResult:
    """Results container for ORACLE-VARX model.

    Attributes:
        forecasts: Selected forecasts using optimal alpha and lag, shape (n_assets, n_days)
        forecasts_all: All forecasts for each alpha, shape (n_assets, n_days, n_alphas)
        p_optimal_all: Optimal lag order for each day and alpha, shape (n_days, n_alphas)
        alpha_optimal: Optimal alpha for each day, shape (n_days,)
        p_optimal: Optimal lag order for each day (at optimal alpha), shape (n_days,)
        alpha_grid: List of alpha values considered
        coefficients_all: Coefficients for all alphas, shape (n_days, n_alphas, p_max, n_assets, n_assets)
        asset_names: List of asset names
        confounder_names: List of confounder names (always non-empty for ORACLE)
        dates: List of date strings for each forecast day
        SE_all: Optional standard errors for all alphas, shape (n_days, p_max, n_assets*p_max, n_assets)
    """
    forecasts: torch.Tensor
    forecasts_all: torch.Tensor
    p_optimal_all: torch.Tensor
    alpha_optimal: torch.Tensor
    p_optimal: torch.Tensor
    alpha_grid: List[float]
    coefficients_all: torch.Tensor
    asset_names: List[str]
    confounder_names: List[str]
    dates: List[str]
    SE_all: Optional[torch.Tensor] = None

    @property
    def method(self) -> str:
        """Returns model type: 'ORACLE-VARX'."""
        return "ORACLE-VARX"

    @property
    def n_alphas(self) -> int:
        """Returns number of alpha values in grid."""
        return len(self.alpha_grid)

    @property
    def coefficients(self) -> torch.Tensor:
        """Returns coefficients indexed by optimal alpha for each day.

        Returns:
            Tensor with shape (n_days, p_max, n_assets, n_assets)
        """
        # For each day, select the coefficients corresponding to the optimal alpha
        n_days = self.coefficients_all.shape[0]
        p_max = self.coefficients_all.shape[2]
        n_assets = self.coefficients_all.shape[3]

        # Initialize output tensor
        result = torch.zeros(n_days, p_max, n_assets, n_assets,
                           device=self.coefficients_all.device,
                           dtype=self.coefficients_all.dtype)

        # For each day, extract coefficients at optimal alpha
        for day_idx in range(n_days):
            alpha_idx = self.alpha_optimal[day_idx].long().item()
            result[day_idx] = self.coefficients_all[day_idx, alpha_idx]

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecasts to DataFrame with dates as index and assets as columns.

        Returns:
            DataFrame with shape (n_days, n_assets)
        """
        return pd.DataFrame(
            self.forecasts.cpu().numpy().T,  # Transpose to (n_days, n_assets)
            index=self.dates,
            columns=self.asset_names
        )

    def get_leadlag_matrix(self, day_idx: int, lag: int, alpha_idx: Optional[int] = None) -> pd.DataFrame:
        """Extract lead-lag coefficient matrix for a specific day, lag, and alpha.

        Args:
            day_idx: Index of the forecast day
            lag: Lag order (1-indexed)
            alpha_idx: Index into alpha_grid (default: None, uses optimal alpha for the day)

        Returns:
            DataFrame with shape (n_assets, n_assets) where entry [i, j] represents
            the effect of asset j at the given lag on asset i
        """
        p_max = self.coefficients_all.shape[2]

        if not 1 <= lag <= p_max:
            raise ValueError(f"Lag must be in range [1, {p_max}], got {lag}")
        if not 0 <= day_idx < len(self.dates):
            raise ValueError(f"Day index must be in range [0, {len(self.dates)}), got {day_idx}")

        # Use optimal alpha if not specified
        if alpha_idx is None:
            alpha_idx = self.alpha_optimal[day_idx].long().item()
        else:
            if not 0 <= alpha_idx < self.n_alphas:
                raise ValueError(f"Alpha index must be in range [0, {self.n_alphas}), got {alpha_idx}")

        # coefficients_all shape: (n_days, n_alphas, p_max, n_assets, n_assets)
        # lag is 1-indexed, so we use lag-1 for 0-indexed access
        coef_matrix = self.coefficients_all[day_idx, alpha_idx, lag - 1].cpu().numpy()

        return pd.DataFrame(
            coef_matrix,
            index=self.asset_names,
            columns=self.asset_names
        )

    def save(self, path: str) -> None:
        """Save results to disk using torch.save.

        Args:
            path: File path to save to (typically with .pt or .pth extension)
        """
        save_dict = {
            'forecasts': self.forecasts,
            'forecasts_all': self.forecasts_all,
            'p_optimal_all': self.p_optimal_all,
            'alpha_optimal': self.alpha_optimal,
            'p_optimal': self.p_optimal,
            'alpha_grid': self.alpha_grid,
            'coefficients_all': self.coefficients_all,
            'asset_names': self.asset_names,
            'confounder_names': self.confounder_names,
            'dates': self.dates,
            'SE_all': self.SE_all,
            'result_type': 'ORACLEVARXResult'
        }
        torch.save(save_dict, path)

    @staticmethod
    def load(path: str, device: Optional[torch.device] = None) -> 'ORACLEVARXResult':
        """Load results from disk.

        Args:
            path: File path to load from
            device: Device to load tensors to (default: None, keeps original device)

        Returns:
            ORACLEVARXResult instance
        """
        if device is None:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=device)

        if checkpoint.get('result_type') != 'ORACLEVARXResult':
            raise ValueError(f"File does not contain ORACLEVARXResult data")

        return ORACLEVARXResult(
            forecasts=checkpoint['forecasts'],
            forecasts_all=checkpoint['forecasts_all'],
            p_optimal_all=checkpoint['p_optimal_all'],
            alpha_optimal=checkpoint['alpha_optimal'],
            p_optimal=checkpoint['p_optimal'],
            alpha_grid=checkpoint['alpha_grid'],
            coefficients_all=checkpoint['coefficients_all'],
            asset_names=checkpoint['asset_names'],
            confounder_names=checkpoint['confounder_names'],
            dates=checkpoint['dates'],
            SE_all=checkpoint.get('SE_all', None)
        )
