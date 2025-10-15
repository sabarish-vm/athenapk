import numpy as np
from matplotlib.axes import Axes
from numpy import float64, typing as npt
import pathlib
from natsort import os_sorted
import typing as tp
import yt
import xarray as xr
from joblib import Parallel, delayed


class Profiles:
    def __init__(self, dirpath: str) -> None:
        self.dirpath: pathlib.Path = pathlib.Path(dirpath)
        self.files: tp.List[pathlib.Path] = os_sorted(
            [i for i in self.dirpath.glob("parthenon.prim.*.phdf")]
        )

    def __load_single(
        self, path: pathlib.Path
    ) -> tp.Tuple[
        npt.NDArray[np.float64], tp.List[str], float, npt.NDArray[np.float64]
    ]:
        units_override = {
            "length_unit": (1.32712440000000e15, "cm"),
            "time_unit": (1.32712440000000e09, "s"),
            "mass_unit": (2.33741002331213e27, "g"),
        }

        ds = yt.load(path, units_override=units_override)  # pyright: ignore
        dd = ds.all_data()
        distances = np.array(dd.fcoords[:, 0])
        prof = (
            (dd.to_dataframe(["prim_velocity_1", "density"]))
            .rename({"prim_velocity_1": "ur", "density": "rho"})
            .to_numpy()
        )
        return prof, ["ur", "rho"], float(ds.current_time), distances

    def load_all(self):
        d0, fields, _t, distances = self.__load_single(self.files[0])

        def job(p) -> tp.Tuple[float, npt.NDArray[float64]]:
            dp, _, _t, _ = self.__load_single(p)
            return _t, dp

        resP = list(
            Parallel(n_jobs=4, prefer="processes")(delayed(job)(x) for x in self.files)
        )
        d0 = np.stack([d0] + [i[1] for i in resP if i is not None], axis=0)
        times = [_t] + [i[0] for i in resP if i is not None]

        data = xr.DataArray(
            d0,
            dims=["t", "r", "F"],
            coords={"t": times, "r": distances, "F": fields},
            name="profile",
        )
        self.data = data


class Helper:
    @staticmethod
    def split_by_sign(
        source: npt.NDArray, destination: tp.List[npt.NDArray]
    ) -> tp.List[npt.NDArray]:
        sign_positive = source > 0
        sign_changes = np.diff(sign_positive).astype(bool)
        split_indices = np.where(sign_changes)[0] + 1
        res = []
        assert type(destination) is list, (
            "destination arg must be a *list* of arrays which needs to be split, if a single array needs to be passed pass it like [array]"
        )
        for i in destination:
            res.append(np.split(i, split_indices))
        return res

    @staticmethod
    def plot_abs_log(
        ax: Axes, x: npt.NDArray, y: npt.NDArray, col: str = "blue"
    ) -> None:
        [xs, ys] = Helper.split_by_sign(y, [x, y])
        assert len(xs) == len(ys)
        for xi, yi in zip(xs, ys):
            sgn = -1 if yi[0] < 0 else 1
            ax.plot(xi, yi * sgn, ls="-" if sgn < 0 else "--", color=col)
        ax.set_xscale("log")
        ax.set_yscale("log")


if __name__ == "__main__":
    prof = Profiles("./")
