import hashlib

from cm1core.cuda_memory_inventory import shape_bytes
from cm1core.cuda_psolver3_terrain_open import (
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE,
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
    psolver3_terrain_open_source,
    terrain_open_acoustic_workspace_shapes,
)


def test_default_kernel_source_pin_remains_exact_and_live_elimination_is_retained():
    original = psolver3_terrain_open_source()
    assert original == CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE
    assert hashlib.sha256(original.encode()).hexdigest() == CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256
    compact = psolver3_terrain_open_source(retain_column_diagnostics=False)
    # 7b444d1b... until the compact core also dropped the stores nothing reads
    # (tests/test_acoustic_compact_dead_stores.py proves the words unchanged).
    assert hashlib.sha256(compact.encode()).hexdigest() == "e27d50c130713ed172ed427c13e961805e5ecc0084952ac8dfe33013f8019b48"
    for access in ("aa0[p]", "bb0[p]", "cc0[p]", "dd0[p]", "piv[p]", "res[p]", "divh[", "hadvp[", "dum3["):
        assert access not in compact
    for live in (
        "ec[p]=m32(-cc,inv)",
        "ed[p]=m32((k==2u)?dd:s32(dd,m32(aa,ed[pb])),inv)",
        "w[idx3(k,y,x,nyw,nxw)]=m32(s32(dd,m32(aa,ed[pb])),inv)",
    ):
        assert original.count(live) == compact.count(live) == 1
    assert original.count("pp3d[p]=up;float tend=m32(kdiv,s32(up,old));ec[p]=tend;ppd[p]=a32(up,tend)") == 1
    assert compact.count("pp3d[p]=up;float tend=m32(kdiv,s32(up,old));ppd[p]=a32(up,tend)") == 1


def test_boundary_workspace_is_fully_priced_including_four_compact_wind_volumes():
    shapes = terrain_open_acoustic_workspace_shapes(400, 400, 237)
    assert tuple(shapes) == ("pua", "pu", "pva", "pv", "pruf", "prvf", "prw", "pre", "prs", "prn", "tous", "tovs")
    assert shapes["pua"] == (237, 402, 403)
    assert shapes["pva"] == (237, 403, 402)
    assert sum(shape_bytes(shape) for shape in shapes.values()) == 618_893_176
