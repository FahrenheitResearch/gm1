"""The namelist grammar CM1 users write, read through the native parser.

gfortran's NAMELIST input takes several assignments per line, blanks as
separators, a missing final comma, upper case, every Fortran logical
spelling, the ``&end`` terminator and ``!`` inside quoted strings.  A file
that CM1 reads must read here too, and must mean the same thing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cm1import_support import namelist_tool, write
from cm1core.cm1import import namelist


@pytest.fixture(scope="module")
def tool() -> Path:
    return namelist_tool()


def parsed(tool: Path, tmp_path: Path, text: str) -> namelist.ParsedNamelist:
    return namelist.inspect(write(tmp_path, text), tool=tool)


def test_two_assignments_on_one_line(tool: Path, tmp_path: Path) -> None:
    for text in (
        " &param0\n nx = 20, ny = 30,\n /\n",
        " &param0\n nx=20,ny=30\n /\n",
        " &param0\n nx = 20  ny = 30\n /\n",
    ):
        values = parsed(tool, tmp_path, text).values()["param0"]
        assert (values["nx"], values["ny"]) == (20, 30), text


def test_upper_case_groups_and_keys(tool: Path, tmp_path: Path) -> None:
    result = parsed(tool, tmp_path, " &PARAM0\n NX = 20, Ny = 30,\n /\n &Param1\n DX = 500.0 /\n")
    assert result.values() == {"param0": {"nx": 20, "ny": 30}, "param1": {"dx": 500.0}}


@pytest.mark.parametrize("spelling, expected", [
    (".TRUE.", True), (".true.", True), ("T", True), ("t", True), (".T.", True), ("true", True),
    (".FALSE.", False), ("F", False), (".f.", False),
])
def test_dot_true_and_t_are_the_same_logical(tool: Path, tmp_path: Path, spelling: str, expected: bool) -> None:
    result = parsed(tool, tmp_path, f" &param0\n terrain_flag = {spelling},\n procfiles = F\n /\n")
    assert result.values()["param0"] == {"terrain_flag": expected, "procfiles": False}
    # The spelling does not reach the canonical document.
    reference = parsed(tool, tmp_path, f" &param0\n terrain_flag = {'.true.' if expected else '.false.'}, procfiles = .false., /\n")
    assert result.document_sha256 == reference.document_sha256


def test_ampersand_end_terminates_a_group(tool: Path, tmp_path: Path) -> None:
    result = parsed(tool, tmp_path, " &param0\n nx = 20,\n &end\n &param1\n dx = 500.0,\n &END\n")
    assert result.values() == {"param0": {"nx": 20}, "param1": {"dx": 500.0}}


def test_exclamation_mark_inside_a_string_is_not_a_comment(tool: Path, tmp_path: Path) -> None:
    # r21.1 declares no string key, so the string rides on an r19 key the
    # importer reports: output_path.  What follows it on the line must still
    # be read, and the real comment after that must not be.
    result = parsed(
        tool, tmp_path,
        " &param9\n output_path = './run!1/', output_th = 1 ! output_u = 0\n output_u = 1\n /\n",
    )
    assert result.values() == {"param9": {"output_th": 1, "output_u": 1}}
    [row] = result.unknown
    assert (row.group, row.key, row.text, row.value, row.value_type) == (
        "param9", "output_path", "'./run!1/'", "./run!1/", "string"
    )


def test_a_missing_trailing_comma(tool: Path, tmp_path: Path) -> None:
    for text in (
        " &param0\n nx = 20\n ny = 30\n /\n",
        " &param0\n nx = 20\n ny = 30 /\n",
        " &param0\n nx = 20, ny = 30/\n",
        " &param0\r\n nx = 20\r\n ny = 30\r\n /\r\n",
    ):
        assert parsed(tool, tmp_path, text).values() == {"param0": {"nx": 20, "ny": 30}}, repr(text)


def test_reals_come_back_as_the_literal_the_file_wrote(tool: Path, tmp_path: Path) -> None:
    result = parsed(tool, tmp_path, " &param3\n fcor = 5.0e-5, zd = 15000, kdiv = 0.10 /\n &param1 tapfrq = 9.0d2 /\n")
    values = result.values()
    assert values["param3"] == {"fcor": 5e-5, "zd": 15000.0, "kdiv": 0.1}
    assert values["param1"] == {"tapfrq": 900.0}
    assert result.groups["param3"]["fcor"].type == "real32"
    assert result.groups["param3"]["fcor"].text == "5.0e-5"


def test_undeclared_keys_are_reported_not_refused(tool: Path, tmp_path: Path) -> None:
    result = parsed(
        tool, tmp_path,
        " &param1\n dx = 1000.0, cfl_limit = 1.0,\n /\n &param2\n pertcor = 1, ptype = 5\n /\n"
        " &param23\n timeavg_num = 5 /\n",
    )
    assert result.values() == {"param1": {"dx": 1000.0}, "param2": {"ptype": 5}}
    assert [(r.group, r.key, r.value, r.group_declared) for r in result.unknown] == [
        ("param1", "cfl_limit", 1.0, True),
        ("param2", "pertcor", 1, True),
        ("param23", "timeavg_num", 5, False),
    ]


@pytest.mark.parametrize("text, where", [
    (" &param0\n nx = 20, nx = 30 /\n", "param0.nx is assigned more than once"),
    (" &param0\n nx = .true. /\n", "param0.nx must be a scalar decimal integer"),
    (" &param0\n terrain_flag = 1 /\n", "param0.terrain_flag must be a Fortran logical"),
    (" &param0\n nx = 20\n", "missing its '/' terminator"),
])
def test_a_malformed_file_is_refused_with_its_place(tool: Path, tmp_path: Path, text: str, where: str) -> None:
    with pytest.raises(namelist.NamelistError, match=where):
        parsed(tool, tmp_path, text)


def test_every_shipped_r21_deck_reads_with_nothing_undeclared(tool: Path) -> None:
    from cm1import_support import RUN_DEFAULT, SQUALL_LINE, SUPERCELL

    for deck in (SUPERCELL, SQUALL_LINE, RUN_DEFAULT):
        result = namelist.inspect(deck, tool=tool)
        assert result.unknown == (), deck.name
        assert result.values()["param2"]["ptype"] == 5


def test_text_outside_the_groups_is_passed_over_and_reported(tool: Path, tmp_path: Path) -> None:
    # gfortran looks for '&name' and passes over everything else, so a title
    # line, a note between groups, words after a '/' and a stray '&end' never
    # stop CM1.  They are read here the same way, and each line is returned so
    # the importer can say CM1 ignores it.
    text = (
        "supercell test at 250 m\n"
        " &param0\n nx = 20, ny = 30\n /\n"
        "ny below was 40\n"
        " &param2\n ptype = 7\n / end of param2\n"
        " &end\n"
    )
    result = parsed(tool, tmp_path, text)
    assert result.values() == {"param0": {"nx": 20, "ny": 30}, "param2": {"ptype": 7}}
    assert [(row.line, row.text) for row in result.outside] == [
        (1, "supercell test at 250 m"),
        (5, "ny below was 40"),
        (8, "end of param2"),
        (9, "&end"),
    ]
    assert result.group_order == ("param0", "param2")
    clean = parsed(tool, tmp_path, " &param0\n nx = 20, ny = 30\n /\n &param2\n ptype = 7\n /\n")
    assert result.document_sha256 == clean.document_sha256
    assert clean.outside == ()


def test_group_order_is_the_order_the_file_opens_them(tool: Path, tmp_path: Path) -> None:
    result = parsed(tool, tmp_path, " &param8\n var1 = 1.0 /\n &param99\n foo = 1 /\n &param0\n nx = 20 /\n")
    assert result.group_order == ("param8", "param99", "param0")


#: A few r21.1 groups declared with CM1's own types (input.F), read the way
#: param.F reads them: each group after a rewind, with end= so a missing group
#: is skipped.  It prints every value, so the parser's reading can be held to
#: the compiler CM1 users build with.
_FORTRAN_READER = """\
program read_groups
  implicit none
  integer :: nx, ny, ptype, ios
  logical :: terrain_flag, procfiles
  real :: dx, fcor
  double precision :: tapfrq
  character(len=512) :: fname
  namelist /param0/ nx, ny, terrain_flag, procfiles
  namelist /param1/ dx, tapfrq
  namelist /param2/ ptype
  namelist /param3/ fcor
  nx = -777; ny = -777; ptype = -777; terrain_flag = .false.; procfiles = .true.
  dx = -777.0; tapfrq = -777.0d0; fcor = -777.0
  call get_command_argument(1, fname)
  open(unit=20, file=trim(fname), form='formatted', status='old', access='sequential')
  read(20, nml=param0, iostat=ios); if (ios > 0) stop 2
  rewind(20)
  read(20, nml=param1, iostat=ios); if (ios > 0) stop 2
  rewind(20)
  read(20, nml=param2, iostat=ios); if (ios > 0) stop 2
  rewind(20)
  read(20, nml=param3, iostat=ios); if (ios > 0) stop 2
  write(*,'(a,i0)') 'nx ', nx
  write(*,'(a,i0)') 'ny ', ny
  write(*,'(a,l1)') 'terrain_flag ', terrain_flag
  write(*,'(a,l1)') 'procfiles ', procfiles
  write(*,'(a,es24.16)') 'dx ', dx
  write(*,'(a,es24.16)') 'tapfrq ', tapfrq
  write(*,'(a,i0)') 'ptype ', ptype
  write(*,'(a,es24.16)') 'fcor ', fcor
end program
"""

_AGREED_WITH_GFORTRAN = {
    "two per line, commas": " &param0\n nx = 20, ny = 30,\n /\n",
    "two per line, blanks": " &param0\n nx = 20  ny = 30\n /\n",
    "value on the next line": " &param0\n nx =\n 20, ny = 30\n /\n",
    "upper case": " &PARAM0\n NX = 20, Ny = 30, TERRAIN_FLAG = .TRUE.\n /\n",
    "T and F": " &param0\n terrain_flag = T, procfiles = F\n /\n",
    "true and .f.": " &param0\n terrain_flag = true, procfiles = .f.\n /\n",
    "&end": " &param0\n nx = 20\n &end\n &param1 dx = 500.0 &END\n",
    "$ groups": " $param0\n nx = 20\n $end\n",
    "! in a comment": " &param0\n nx = 20 ! the x size, ny = 99\n ny = 30\n /\n",
    "no trailing comma": " &param0\n nx = 20\n ny = 30\n /\n",
    "CRLF": " &param0\r\n nx = 20,\r\n ny = 30\r\n /\r\n",
    "tabs": " &param0\n\tnx\t=\t20,\tny = 30\n /\n",
    "real spellings": " &param1\n dx = 5.e2, tapfrq = 9D2\n /\n &param3 fcor = .5E-4 /\n",
    "integer for a real": " &param1\n dx = 500, tapfrq = 900\n /\n",
    "signs": " &param0\n nx = +20 /\n &param1\n dx = -0.0\n /\n",
    "groups in any order": " &param2\n ptype = 7\n /\n &param0\n nx = 20\n /\n",
    "text outside groups": "my run\n &param0\n nx = 20\n / words\n notes\n &param2 ptype = 7 /\n",
    "an undeclared group": " &param0\n nx = 20\n /\n &param99\n foo = 1\n /\n &param2 ptype = 7 /\n",
}


@pytest.fixture(scope="module")
def fortran_reader(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import shutil
    import subprocess

    if shutil.which("gfortran") is None:
        pytest.skip("the gfortran comparison needs gfortran")
    work = tmp_path_factory.mktemp("gfortran-namelist")
    (work / "read_groups.f90").write_text(_FORTRAN_READER, encoding="ascii")
    subprocess.run(["gfortran", "-O0", "-o", "read_groups", "read_groups.f90"], cwd=work, check=True, capture_output=True)
    return work / "read_groups"


@pytest.mark.parametrize("text", list(_AGREED_WITH_GFORTRAN.values()), ids=list(_AGREED_WITH_GFORTRAN))
def test_the_parser_reads_what_gfortran_reads(tool: Path, tmp_path: Path, fortran_reader: Path, text: str) -> None:
    import math
    import struct
    import subprocess

    path = tmp_path / "namelist.input"
    path.write_bytes(text.encode("ascii"))
    completed = subprocess.run([str(fortran_reader), str(path)], capture_output=True, text=True, check=True)
    fortran = dict(line.split(None, 1) for line in completed.stdout.splitlines())
    ours = namelist.inspect(path, tool=tool).flat()
    for key, printed in fortran.items():
        if key in ("terrain_flag", "procfiles"):
            unset = {"terrain_flag": False, "procfiles": True}[key]
            assert ours.get(key, unset) is (printed.strip() == "T"), key
        elif key in ("nx", "ny", "ptype"):
            assert ours.get(key, -777) == int(printed), key
        else:
            value = ours.get(key, -777.0)
            if key != "tapfrq":  # CM1 stores the others as default REAL
                value = struct.unpack("<f", struct.pack("<f", value))[0]
            expected = float(printed)
            assert value == expected and math.copysign(1.0, value) == math.copysign(1.0, expected), key


@pytest.mark.parametrize("text, where", [
    (" &param0\n nx = , ny = 30\n /\n", "param0.nx has an empty value"),
    (" &param0\n nx = 1*20\n /\n", "param0.nx must be a scalar decimal integer"),
    (" &param0\n nx = 20\n /\n &param0\n nx = 30\n /\n", "appears more than once"),
])
def test_what_gfortran_reads_but_the_parser_refuses_is_refused_by_name(
    tool: Path, tmp_path: Path, text: str, where: str
) -> None:
    # gfortran reads a null value (the key keeps its value), a repeat count
    # and a repeated group (the first one wins).  None appears in CM1's decks
    # or the scoping corpus; each is refused here with its place rather than
    # guessed at, and the user can write the plain form.
    with pytest.raises(namelist.NamelistError, match=where):
        parsed(tool, tmp_path, text)
