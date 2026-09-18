from __future__ import annotations

import numpy as np

_NUMPY_TYPE = {("F", 4): "f4", ("F", 8): "f8", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4",
               ("U", 1): "u1", ("U", 2): "u2", ("U", 4): "u4"}


def parse_pcd(data: bytes) -> np.ndarray:
    """PCD bytes (as returned by Viam cameras and vision services) -> Nx3 float points in MILLIMETERS.

    Handles ascii and binary PCDs with any extra fields (rgb etc.). Viam writes PCD
    files in meters while the rest of its API is in mm; the unit is detected from the
    data (nothing a wrist camera looks at is 20 m away, or closer than 20 mm).
    """
    header: dict[str, list[str]] = {}
    offset = 0
    while True:
        end = data.index(b"\n", offset)
        line = data[offset:end].decode("ascii", "replace").strip()
        offset = end + 1
        if line and not line.startswith("#"):
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break

    fields = header["FIELDS"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    n = int(header["POINTS"][0]) if "POINTS" in header else int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    kind = header["DATA"][0].lower()
    if n == 0:
        return np.zeros((0, 3))

    if kind == "ascii":
        table = np.loadtxt(data[offset:].decode("ascii").splitlines(), ndmin=2)
        cols = np.cumsum([0] + counts)
        xyz = np.column_stack([table[:, cols[fields.index(a)]] for a in "xyz"])
    elif kind == "binary":
        dtype = np.dtype([
            (name, _NUMPY_TYPE[(t, int(s))], (c,) if c > 1 else ())
            for name, t, s, c in zip(fields, header["TYPE"], header["SIZE"], counts)
        ])
        records = np.frombuffer(data, dtype=dtype, count=n, offset=offset)
        xyz = np.column_stack([records[a].astype(np.float64) for a in "xyz"])
    else:
        raise ValueError(f"unsupported PCD encoding {kind!r}")

    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) and np.abs(xyz).max() < 20.0:
        xyz = xyz * 1000.0  # meters -> mm
    return xyz
