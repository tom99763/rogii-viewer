"""GR-mismatch heatmap computation — the hengck23-style visualisation.

For a horizontal well + its typewell, compute:
  * a per-segment-compressed GR misfit heatmap (typewell rows × HW segments)
  * the ground-truth typewell-row index per HW segment (when TVT is available)
  * a DTW alignment path through the heatmap (when dtw-python is available)

All knobs follow hengck23's notebook conventions but are tunable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

try:
    from dtw import dtw, rabinerJuangStepPattern  # type: ignore
    HAS_DTW = True
except Exception:
    HAS_DTW = False


@dataclass
class HeatmapBundle:
    """Everything the viewer needs to render the GR-mismatch heatmap tab."""
    heatmap: np.ndarray            # (n_tw, n_hw) GR difference: hw_seg - tw_seg
    h_seg_gr: np.ndarray           # (n_hw,) smoothed HW GR per segment
    h_seg_md: np.ndarray           # (n_hw,) center MD per segment
    h_seg_tvt: np.ndarray          # (n_hw,) center TVT per segment (NaN where unknown)
    t_seg_gr: np.ndarray           # (n_tw,) typewell GR sample
    t_seg_tvt: np.ndarray          # (n_tw,) typewell TVT axis
    gt_path: Optional[np.ndarray]  # (n_hw,) ground-truth typewell-row index per HW seg, or None
    dtw_path: Optional[np.ndarray] # (n_hw,) DTW typewell-row index per HW seg, or None
    ps_seg_index: int              # HW segment index containing PS
    S: int                         # compression: ft per segment

    @property
    def n_hw(self) -> int:
        return self.heatmap.shape[1]

    @property
    def n_tw(self) -> int:
        return self.heatmap.shape[0]


def compute_gr_heatmap(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    ps_idx: int,
    *,
    S: int = 32,
    n_before: int = 8,
    n_after: int = 16,
    tw_window: int = 32,
    gr_smooth_win: int = 51,
    gr_smooth_poly: int = 2,
    run_dtw: bool = True,
) -> HeatmapBundle:
    """Build the heatmap bundle for one well centred at its PS.

    Args:
        hw, tw: horizontal-well and typewell DataFrames (as in the dataset).
        ps_idx: row index in `hw` where TVT_input first becomes NaN.
        S: compression — average GR over S consecutive HW rows per segment.
        n_before, n_after: segments to extract before / after PS (HW axis).
        tw_window: rows of typewell to include either side of the PS-equivalent
            typewell index (TW axis half-width).
        gr_smooth_win, gr_smooth_poly: Savitzky-Golay smoothing of HW GR.
        run_dtw: when True and dtw-python is available, also compute the DTW
            alignment path through the heatmap.
    """
    if "MD" not in hw.columns or "GR" not in hw.columns:
        raise ValueError("hw missing MD/GR columns")
    md = hw["MD"].to_numpy()
    gr_hw_raw = hw["GR"].to_numpy()

    # Fill NaN HW GR by neighbour interpolation, then smooth
    s = pd.Series(gr_hw_raw)
    gr_hw_filled = s.interpolate(limit_direction="both").bfill().ffill().to_numpy()
    # savgol_filter needs odd window length and window <= len
    win = min(gr_smooth_win, len(gr_hw_filled))
    if win % 2 == 0:
        win -= 1
    win = max(win, gr_smooth_poly + 2)
    if win % 2 == 0:
        win += 1
    gr_hw_smooth = savgol_filter(gr_hw_filled, win, gr_smooth_poly)

    # Anchor the typewell window at the typewell row whose TVT is closest to
    # the last-known HW TVT (TVT_input[ps_idx-1]).
    if "TVT_input" in hw.columns and ps_idx > 0:
        tvt_anchor = float(hw["TVT_input"].iloc[ps_idx - 1])
    elif "TVT" in hw.columns and ps_idx > 0:
        tvt_anchor = float(hw["TVT"].iloc[ps_idx - 1])
    else:
        tvt_anchor = float(tw["TVT"].iloc[len(tw) // 2])

    tw_tvt_arr = tw["TVT"].to_numpy()
    tw_gr_arr = tw["GR"].to_numpy()
    t_ps = int(np.argmin(np.abs(tw_tvt_arr - tvt_anchor)))
    j0 = max(t_ps - tw_window, 0)
    j1 = min(t_ps + tw_window, len(tw))
    t_seg_gr = tw_gr_arr[j0:j1].copy()
    t_seg_tvt = tw_tvt_arr[j0:j1].copy()

    # HW segment range around PS
    i_start = max(ps_idx - n_before * S, 0)
    i_end = min(ps_idx + n_after * S, len(hw))
    # Trim to a whole multiple of S
    span = (i_end - i_start) // S * S
    i_end = i_start + span
    if span < S:
        raise ValueError(f"not enough rows around PS to form a single segment (S={S})")
    n_hw = span // S

    seg_view_gr = gr_hw_smooth[i_start:i_end].reshape(n_hw, S)
    seg_view_md = md[i_start:i_end].reshape(n_hw, S)
    h_seg_gr = seg_view_gr.mean(axis=1)
    h_seg_md = seg_view_md.mean(axis=1)
    if "TVT" in hw.columns:
        seg_view_tvt = hw["TVT"].to_numpy()[i_start:i_end].reshape(n_hw, S)
        h_seg_tvt = seg_view_tvt.mean(axis=1)
    else:
        h_seg_tvt = np.full(n_hw, np.nan)

    # PS in segment coordinates
    ps_seg_index = (ps_idx - i_start) // S

    # heatmap[i, j] = h_seg_gr[j] - t_seg_gr[i]
    heatmap = h_seg_gr[None, :] - t_seg_gr[:, None]

    # Ground-truth path: for each HW segment, find the typewell row whose TVT
    # is closest to that segment's mean TVT.
    if np.isfinite(h_seg_tvt).all():
        gt_path = np.array(
            [int(np.argmin(np.abs(t_seg_tvt - t))) for t in h_seg_tvt],
            dtype=np.int32,
        )
    else:
        gt_path = None

    # DTW path: align HW GR sequence onto typewell GR sequence using
    # rabinerJuangStepPattern(1, "c") — the same step pattern hengck23 uses.
    dtw_path: Optional[np.ndarray] = None
    if run_dtw and HAS_DTW and len(h_seg_gr) >= 3 and len(t_seg_gr) >= 3:
        try:
            alignment = dtw(
                h_seg_gr.astype(float),
                t_seg_gr.astype(float),
                keep_internals=True,
                step_pattern=rabinerJuangStepPattern(1, "c"),
            )
            # For each HW segment index, take the first matched TW row index
            path = np.full(n_hw, -1, dtype=np.int32)
            for hw_i, tw_i in zip(alignment.index1, alignment.index2):
                if path[hw_i] < 0:
                    path[hw_i] = tw_i
            # Fill any uncovered HW indices via nearest-neighbour
            for k in range(n_hw):
                if path[k] < 0:
                    nearest = next(
                        (path[m] for m in range(k - 1, -1, -1) if path[m] >= 0),
                        None,
                    )
                    if nearest is None:
                        nearest = next(
                            (path[m] for m in range(k + 1, n_hw) if path[m] >= 0),
                            0,
                        )
                    path[k] = nearest
            dtw_path = path
        except Exception:
            dtw_path = None

    return HeatmapBundle(
        heatmap=heatmap.astype(np.float32),
        h_seg_gr=h_seg_gr.astype(np.float32),
        h_seg_md=h_seg_md.astype(np.float32),
        h_seg_tvt=h_seg_tvt.astype(np.float32),
        t_seg_gr=t_seg_gr.astype(np.float32),
        t_seg_tvt=t_seg_tvt.astype(np.float32),
        gt_path=gt_path,
        dtw_path=dtw_path,
        ps_seg_index=int(ps_seg_index),
        S=int(S),
    )


def map_path_to_tvt(path: np.ndarray, t_seg_tvt: np.ndarray) -> np.ndarray:
    """Convert a path of typewell-row indices into TVT values."""
    return t_seg_tvt[np.clip(path, 0, len(t_seg_tvt) - 1)]
