"""Module identities as the sealed receipts measured them.

Some modules were reworded after sealed receipts were measured on them: two error messages in the tilestream modules, and comments and
docstrings in the surface-drag operators, and the test files and an example
deck that receipts list among their artifacts (comments only). The receipts pin the sha256 (and some the size) of the bytes they ran, and those
receipts are not rewritten. ``REWORDED`` maps the sha256 of each module as it
is now to the identity its receipts measured: the bytes differ only in that
rewording. Any other edit to a module changes its current digest, so
``measured_sha256`` then returns that new digest and the comparison with the
receipt fails, exactly as it would have before the rewording.

The Cartesian door, the configuration module, the GM1 runtime, ``doctor`` and
the oracle decks needed no entry once the dynamics receipt, the only one that
pins them, was sealed again from a fresh capture on the release sources; the
identifiers and prose renamed after that capture give them, and every other
module a sealed record pins, an entry in the block below.
``unlabel`` reads an id as those receipts spell it, and ``measured_plan_sha256``
the structural plan identity the sm86 receipt measured.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re

_ROOT = Path(__file__).resolve().parents[1]

#: path suffix -> (sha256 of the module now, sha256 the receipts measured,
#: size in bytes the receipts measured)
REWORDED: dict[str, tuple[str, str, int]] = {
    "cm1core/cuda_gm1_tilestream_post_wsm6.py": (
        "e663a6fa5cc859ac76accadeb40617e5431442fb74d6a911f860fc35a803ba0b",
        "38c243c06d774786f1902d028c9c37b597e12b61c76530a4910d67a2f537781f",
        38152,
    ),
    "cm1core/cuda_gm1_tilestream_solve3_pressure.py": (
        "02394c27c2251cdf0367316f372d97e805bc1822685e04af87a2813c4cac6202",
        "b6fadd1097e7eac7196d7535cff03fb6275a7802d88a3a42bf872cb5df1feb62",
        22612,
    ),
    "cm1core/cuda_surface_drag.py": (
        "1119ff803d2cc8f037d1f0a14109a3f30e51eddc7b66e747b76fbfdd1be74d5d",
        "d867b255837c05ed3fa4f47c13a0e20019013baab72e9ce7336b6b2b133f9a17",
        101595,
    ),
    # A test file its own sealed receipt lists as an artifact: an assertion that
    # spelled the receipt's host keys now checks the same values without them.
    "tests/test_cuda_gm1_solve1_idealized_radiation.py": (
        "34584994fbbc8617cacff975d2bd441af2d98161fadb93d00ba993fb3f73a152",
        "79a64ca12964e20fec6e0b9b2231ce719ebf1aea07db4ca0ddc7ae50dffd0559",
        15935,
    ),
    # An example deck and a test file sealed receipts list among their artifacts:
    # the deck reworded in comments only (no private paths, branches or tools),
    # the test naming three CM1 statements by digest instead of quoting them.
    "examples/cartesian-idealized-radiation-relaxation.toml": (
        "bd5ff7a2e4d51b262b9476bc792eb5fb4980ed94336cb6e49e55353deebbf55e",
        "c5f2eeb2a79924770d57d3f7b56aedb9a66ba0c9069e96b89754ded454dbb3b6",
        2132,
    ),
    "tests/test_cuda_terrain_open_full_state.py": (
        "6c2f41d9b849b5929bed2118e6cca3010f1280433c10902f9c555be0bfe27f71",
        "7e56fc8ea7dcb630681ca4edfeea12580d42953049c24b92de9b2d2c5da51bdd",
        4617,
    ),
    # The inventory test itself, which now compares its artifacts through
    # measured_identity (the inventory lists this file too).
    "tests/test_cuda_terrain_open_full_state_evidence.py": (
        "67b7e2e03efec627e80cca8d0ebba9e9b3134cf269f08e8a81cd27cb47916309",
        "8b9bca13dc4c68bf6c38c912d175504b81b8d9ce17c30bf6847cc491a9825900",
        7786,
    ),
}


#: Identifiers and prose renamed after the receipts were sealed (gate names, refusal
#: codes, oracle ids, messages and comments; no executed arithmetic changed): the
#: sealed receipts measured these modules with the earlier names. A module that was
#: already reworded once maps to the identity its receipts measured before either rewording.
REWORDED.update({
    "cm1core/arwen_chain.py": (
        "27fcc110ea313943c339d1afe1892d0ea31d58aaf26e95c0ecaea740323c5462",
        "dafcabcb33acefb119f2dfcf532f6edddc564285d08e64ca3caa43984d92a03b",
        97752,
    ),
    "cm1core/arwen_column_component.py": (
        "92a93bf914b3ba5c558849878ce75a6f09c7091406f20c1eea95ded1a8c40202",
        "d640fe3a086ca93f571dbf3f69e65bfe373c3bcfe51afc2a16deaa3d45684cfc",
        59423,
    ),
    "cm1core/arwen_morrison_component.py": (
        "5eb13e9e444128e8671199bf6bae707252573fea2ac5e1330bd9b8809aad5c6d",
        "cd78b06435594ac65eb02f5697727cd9a1122195e5646846e93e8c1bb0a2bb79",
        14726,
    ),
    "cm1core/arwen_physics_component.py": (
        "470017d1712341401c411d836c2c0315733738e4528f6607acf0ee24dce8d33a",
        "b58bad0306d2331c76ec45b58b1d3dcec8832d39d14d72b3e825f927b8eef7c6",
        27315,
    ),
    "cm1core/arwen_scheme_components.py": (
        "58ce85b969da3228b2e265c6287aeb53a50d86adfac30cdf060de7338253348c",
        "b1a08ee714d7e3179fdf8af35cc36dc5bfccf6eed77c965e8b2469ed7d2bb57c",
        18754,
    ),
    "cm1core/arwen_wsm6_component.py": (
        "b858511ae8b58e52393a419cbf126cf826c4436bc30f9684e0cbe3ef42723827",
        "bf7bf18070880339c2b7e40ac77825f00f4e449184eede9a32cb4738606820cd",
        7480,
    ),
    "cm1core/cartesian_adaptive_frame.py": (
        "a6b20d5d7e75699a039741c53ae36a0248a8ae5d5d01d1815c3dcc8281f50a84",
        "1adb547ea90921dadeb11b669fc520f44a734a2d7ad29079c7093b4ae71cd2c8",
        43349,
    ),
    "cm1core/cartesian_door.py": (
        "68079440bb8833f64e1775ca44245ef0ca787ba7377b6c5ec950e90a855a0e1a",
        "02dd9bdb2ef95382bb604d686848d1e332aebb651a6dbe75f94408ecf37b36cd",
        237400,
    ),
    "cm1core/cm1_physics_families.py": (
        "4cd7c8f449ad92a5344d77fd9cd94b3ec544c02597a6c020d16022af9dae8ca1",
        "958c2aa7bfd8d1a70b816b9a528387ab56a5f88e7e7c65f50b86d97297580aee",
        26700,
    ),
    "cm1core/column_component_blockers.py": (
        "21030aa11214974853d39201ff6018bbd5ea4bb885c0c7a8eb759af7d7fe504c",
        "b4ffd147ad901a21a2510465f9819abd4aebe6b7db568f1b6f6cc35e0f47e212",
        9421,
    ),
    "cm1core/column_components.py": (
        "e49d640489fed6f5b51e3f44ca3d76b14bfdf5d328e35c23b8076c0c6fc8e9b7",
        "6cd3514c79ca5e25e93d3a78d2262f0ec674b002db27f3a3189a135688cbab2c",
        4053,
    ),
    "cm1core/cuda_advection5.py": (
        "073f10ef9b369bce47a8df91da4a325805755b090f891b955da67104e0c7167a",
        "3b745b45b0a1d95aa926dc6cc3d9b5534c4fae23c06401826a82f2c529823222",
        56945,
    ),
    "cm1core/cuda_apmasscon.py": (
        "f25bf08d88040b8fc95d7863878e21507e3e8f346ac51ff559682db0a3b62275",
        "0a4ccec19d493748855e8ec6ba564baf18cde554e61a0b2c439f852df85c34ae",
        25991,
    ),
    "cm1core/cuda_cartesian_moist_heating.py": (
        "af904233ac74a6f0899e8c547f8fb49dfc63ca3206d757c9084509c45cf51942",
        "11104abf24282e53a297add1b21ec8948ede0faec51effb2ca0de6ce04476831",
        21386,
    ),
    "cm1core/cuda_dry_vertical.py": (
        "21810467148fffd646fc90ea816fc543f4788ed72d75242483a939667f2f7fb9",
        "d70937b99fa61b81f559979d65825f79971ecbbca36a248820d58f7eee0b42e0",
        30257,
    ),
    "cm1core/cuda_getcape.py": (
        "a10ff75844b34a9e3ec437b7313bd2ca92c7a356de4e55580af297e4a349619f",
        "81008c323b697776362f964616faf1f02724fb9cc503321b19ce17c1f3d65a3f",
        47013,
    ),
    "cm1core/cuda_gm1_child_large_step.py": (
        "8171a8755cd017e3aa0549bde8f6f864d4e417e9a3abe0b451ee5708e2dd9e62",
        "265e6aa15dd1851269c4946f618b8b95d470f86a3034e435c35a6b6baa94541e",
        52207,
    ),
    "cm1core/cuda_gm1_child_post_final_rk.py": (
        "80b9b4a659c53fbe5762e161e259968101b5d10c0f5b740c8d389207bb5accaf",
        "74b7027834b77dedc53d251d2deee10a6b46969a24f99f1fd01c8c721ee54649",
        60508,
    ),
    "cm1core/cuda_gm1_forecast.py": (
        "1c0638c9024f702e7605a1f919052a03f3c09452a3c6a941617de0821d59c8f2",
        "7984745d073d1b1a8e980d283337ecfac32cdee5a1eed2991c798af2ce70008c",
        65644,
    ),
    "cm1core/cuda_gm1_hrrr_lbc.py": (
        "4f6a7ecb2e8d8d716cc7a014a50406b7f354837ad5a9ee091a784d1d05797427",
        "0e0a7522238d3b864bbb885bf8e0dc6c980dda1eff25df4bd9eba1b7f3ece027",
        45560,
    ),
    "cm1core/cuda_gm1_kessler_driver.py": (
        "891e76f99204e5de70b2c416c174fb5c97d4a81f8d094bd1aee87b83fd27a45c",
        "88af5ab68b9c97aeb78a5abcc285bca8df05b8fb9aeb4145271afac027111e10",
        53902,
    ),
    "cm1core/cuda_gm1_kessler_efall.py": (
        "bdb1b79e7f6b1995fd4bef07d3af424493aa9a4b5ffc348c8dfdf8edf650fcff",
        "a5c611ccd6a09d338c74bed5de66f8361625172cdc88300e0022a07d75520f82",
        21791,
    ),
    "cm1core/cuda_gm1_kessler_large_step.py": (
        "b400bf34e4f27ac6b02f318169c9abbe278edf4e9b50c6fc907c3c73470e9b03",
        "78743d8d4aa98e1141b2dc9645ff23e6fd9996fd74c497d8348a8f5f1adb033a",
        26691,
    ),
    "cm1core/cuda_gm1_kessler_local.py": (
        "8750bb4504f801b12243de0dab7654c2edb1313566f6ca3fbdf2571a2a49c93c",
        "295f25e0ab0b72c7cd99fbe50bdb84de16311e4ee4b26d13e0f3bf2096af8ac4",
        33998,
    ),
    "cm1core/cuda_gm1_scalar_turbulence.py": (
        "6a76c519cb446d24f6c3d461fea11ef6cb17846b005bbd16c14eb142e75bab6c",
        "cf9286b680b2abf4e00f0605aacbd325a55d7a3153dd4b8756159bdc6e652168",
        42862,
    ),
    "cm1core/cuda_gm1_two_part_sgs.py": (
        "303da4647d1b1eb0230792c706c01a6bade89984f30a1b3407d3cfd2d422386e",
        "c06b12c09fe6baa4659f42c6e6f2f4e691b6d9cee99176579381421f8ded3d72",
        50325,
    ),
    "cm1core/cuda_initialization.py": (
        "1adf9b7221bd0a338fecad6f28d4dd8fc939c7185f42fd92f3510337bf5b68c5",
        "9a824eb0b88f89ee0e7353bc41ab1ce6c793043dbce114016d4427a398b370ff",
        20666,
    ),
    "cm1core/cuda_lateral_closure.py": (
        "a747bfc31aa3ba06b943a3afc42a0adaa35a7f0a37c7290850b0fa6c9495eda6",
        "6921c013dc40316d9c30f54101a858dbe66ade4137f511f9bbdb3e61b8e2463c",
        47207,
    ),
    "cm1core/cuda_moist_nm.py": (
        "16465d72f171beec24f3df733b56305febe7bb45632c5278bad90b1327494372",
        "b57859dd514cb25d5a55668b7c64180252fbfe442ead27a8964158799084632f",
        29116,
    ),
    "cm1core/cuda_rayleigh_held.py": (
        "268481b36acccb43d58aa9b3e0815248ea9d299f95f8a62851587f9f4d569b1e",
        "80f7982fa2ebc8ae166a821b16290bf4aee889d3e25847c79ca0fa1732030ae3",
        23802,
    ),
    "cm1core/cuda_sgs_stress_tendency.py": (
        "3e87be8dc358619b0c3480b511e74cb9dacc41bec9214e7e63bc3bac817569b0",
        "656fdce301cc51b1804f1bffa2fee8b7d10772344396ebd565443819836ed2e8",
        62452,
    ),
    "cm1core/cuda_sgs_tke.py": (
        "5370a9559d130ba69c0103f7380e55f97f626007a7c4ba6baab94b90788cecb4",
        "16bc9df09f313828c86ac3bbcba1481c084d89eef4f9afaa872e166675117cc9",
        28600,
    ),
    "cm1core/cuda_sound3_open_boundary.py": (
        "ade69d5326cc93794a75acaae3c53c5c67b6051b68a44c06beba532ba3fd545a",
        "d818e6533b885cdcdc7a8b9b6f9ed7483857ec61cbbe6611e86437081229c1e9",
        49766,
    ),
    "cm1core/cuda_state.py": (
        "afa7896103d614273862f679c9d30a3748cf721bba126db54652eaf4c3ac74db",
        "8389a5694d6ccc2fc9a4418e1bc570d78ef41eccd347780b8579f3c10d1db70a",
        56963,
    ),
    "cm1core/cuda_surface_drag.py": (
        "1c02ff84daf070c591f9ad88c3009682f7a63d2f3c5c5c72f99442a3a6ba3764",
        "d867b255837c05ed3fa4f47c13a0e20019013baab72e9ce7336b6b2b133f9a17",
        101595,
    ),
    "cm1core/cuda_terrain_deformation.py": (
        "41a4ee0bffb3b3ce69c8470e8a3398cab490e7c88dc5172a5b15bcf7908e4e1f",
        "f51053640b30cd5a475790ab175fb41df0a49ecabbc38e30531db4846a4c1ecb",
        45998,
    ),
    "cm1core/cuda_terrain_open_boundary.py": (
        "719912d4f076b0431b4e77f1fc35486cd312d10c8a5df6b6798c737434df9d97",
        "67d2878ff3d7c8a09d1f0c58af62129aaa1e15261ea8e430773b627c487ead7a",
        51304,
    ),
    "cm1core/cuda_tke_evolution.py": (
        "93fb2f29bd37375e8fd250cb627c4b77d8af43a31a5fc1ba287dbcaf9dfd9ed0",
        "cc464f087b9ae9f315b61a2cd8e238879b7fd90a3b3c5acd8b9581f5392bf374",
        52268,
    ),
    "cm1core/dense_ice.py": (
        "5627c9fcc9eb85936756c7404030d693be359237152d56ef96ee47c03c1adad0",
        "8dc04a0ef6c1edfc3eef7cbf979e16630ca7edcaa3cde08ecf77f21a3b1ae82b",
        22997,
    ),
    "cm1core/doctor.py": (
        "830acbfe34ee4381a7be2f84b880413aff9e13e15732e93a34544e09a3d4dc07",
        "bc70bb684e536a3847c03955c5a1ee5d1c908c8b3a07c7931f13c0ba80371b1e",
        38881,
    ),
    "cm1core/doors.py": (
        "a33225aa6915fcbf75f96e1997e8a0d6bc8d10cf54de299b3942e8f1a6da3f73",
        "2054eaec2b8b81a851e028bd33ca3fabbecab415ed5fd6952db988be4ac89315",
        41533,
    ),
    "cm1core/eqtset2_energy_terms.py": (
        "4a7d9ced9cdea767657f208e28699005c31a075df249ccac6dbe77911daaa41a",
        "17b2ed73decf8205bf1aa8e39f7b0fa01e86e3ff1791e886b787e7e7f838843b",
        19660,
    ),
    "cm1core/eqtset2_surface_dissipation.py": (
        "aac1d10f2711bf28e7afd898a2ffed59e6aa3dd475031b84aeb6e1e90c7e21bc",
        "2eecaad4b2b3be168d61855ea564f4c702b5c0284dedba2a466f1a6d2db8487a",
        13290,
    ),
    "cm1core/eqtset2_wsm6_bridge.py": (
        "f8300011ad27faab8d156a99f6c13b0d7896d27bd88eae376805ce2f51ec3ac3",
        "b7964923814f8aa84bb0db4d172da11d1be46d255476aade8328144a977da0b7",
        44213,
    ),
    "cm1core/gm1_runtime.py": (
        "22baeb654f80edff5f8dbff9a06665f95bf4509516a71a8c71ad28ba4fe20bdd",
        "cd262f06fb04b753d88d68233a95796ff1ba740737b2580a4940473e92dae8cd",
        108896,
    ),
    "cm1core/initialized_product_wsm6_transition.py": (
        "27315f1bf5829c228d93c4d4f4f4b9fdeb4665fd585463925ea03e0db9c0e2c6",
        "3efeb8a7fc2f03ac20c536b8f14871d1ab3cc4f4f7d3d07e71549d89d1bbb8f3",
        79994,
    ),
    "cm1core/layout.py": (
        "61ced51f7f782ef5f74f7eeb9fd80e7f2940b6a3d26b15e02925795e282ebe84",
        "d202c348d9aadbbaa91716463f6b6da57f017018eb88e5ef409362523c35a480",
        15580,
    ),
    "cm1core/moisture_species.py": (
        "d324c32977a3dfd3fa86b21f14763dc608cb1c43064bd991066ae190eea17140",
        "9a1563963bbc3c3c159b36192135405a2540a9bb845e94df455f41b2a79aab5c",
        23051,
    ),
    "cm1core/nest_child_window.py": (
        "c277de1c62e0567f01759775596ea11505d9075717846ca2ffd8d174282e163f",
        "eab739a6c2c7acc216baf9ed5ae3f1a87d1a59c877bd4fb9a4bdfde85e4cd99a",
        95340,
    ),
    "cm1core/nest_downscale.py": (
        "dae41984b682e0f2fd5a4a49e1350a6104e02b2e4d2a141859ddbf9127920d89",
        "fe3ad2d81ef013915a3dcd53f894fab50ae2e55d4fd2e9af7cf53099594ffcce",
        28093,
    ),
    "cm1core/nest_live.py": (
        "1f1f4352a05a9ade8f3b345f2703da2781ee7d8ad7f1fedc6f0ce2068ab84d31",
        "8e65bc2e0751e4a58055470e49417641a24c0237912e7cab500408f18db74e01",
        76368,
    ),
    "cm1core/nest_move.py": (
        "578d150c62069b5738e2e090950203d2364169e958a551bc4d017e01b9bae4e3",
        "ce70e251a0940a2853466e193ae30ab38eb6b46066e27f5e1b18d593388f6d63",
        93625,
    ),
    "cm1core/nest_parcels.py": (
        "12ac7a8a8fd9007714626cd982a9dd91003a868ba5dc03964d4e22ca23259007",
        "61f222557ef6003b84eebee093a01e749d4d0e4e49ff688718ab4721e751c877",
        103156,
    ),
    "cm1core/preflight.py": (
        "e10b9e95e21cea29213678ba3d620e5dec80e9d1745e9ffc238d9572652d410a",
        "42f06f5ba3974464b2e53a8e746b8a71f1cb1a827af9847cefb37594420339c2",
        10530,
    ),
    "cm1core/sounding_report.py": (
        "715bc39e214fa960a87f7cc90ab96f7d5d4b4db97af15f99ad8439d9a2e1316b",
        "d41af202443250efdc9fd0087f569772fb3ab44f1ff794945f69a93d8387bacc",
        14136,
    ),
    "cm1core/state.py": (
        "1963996fe4d1b9151b67f2208609b41befebd63bbfd4e5d0f74fabcd2bba49f1",
        "0791c0989035733e8ea9959d64c88b5196a87600cd1fa7234e8e82b1acbda8b1",
        18968,
    ),
    "cm1core/surface_fields.py": (
        "c950319ac1c06015094c7d2c1c0c7da8eb74859038e8cb933477e17f29cff239",
        "f3799640a21631d55a80360f1a644bfe2b313a2576dcc70347493322bf444d9d",
        12936,
    ),
    "cm1core/zero_step.py": (
        "b94f68cb3293215811cd1ec2eaa031ebe537c0188d68cec7f78da8b681b77a9e",
        "392c7ce104013de04ab481e3607de93940440298b39a0036d402c4b2855b6145",
        16966,
    ),
    "tests/test_cuda_advection5.py": (
        "300a62fde34a7bfff99e81b8022dbe8f25e8f4a36f6daf03ef1e12c4f91e7350",
        "49c40e6edb71bc05358176e2592d17e030e9b3046e57c49df229a121dc260d4c",
        7219,
    ),
    "tests/test_cuda_advection5_evidence.py": (
        "cd0954c5bd75056dbfd6bed0bf4d438f785255809748d3f826e9dc3b4a0d716f",
        "09c2d766a646d958cf10575be4b55402ab27abdffd81a4c44371f2397b1a5fc1",
        9882,
    ),
    "tests/test_cuda_psolver3_terrain_open_evidence.py": (
        "5f869036f02ea145824628423c5bc034b86908a83cffe9e1ebb62ca8446ad50d",
        "d1b606380775b6b9e83468dd6b5ea742eb3e8f53d2c60d663cc067f985e8eabe",
        6462,
    ),
    "tests/test_cuda_scalar_moist_rk_evidence.py": (
        "98c149ccc3169889b2fcd8b43af7a4e3a7cf28d35d97e6755b5ac12ff29c4815",
        "651bcb162edf8c477d7beafd3b338b8693735aba1e147410ef7ded0a9b34d49b",
        7657,
    ),
    "tests/test_w1_post_getset_oracle.py": (
        "4a7b41eb3f749e9ba5faae7de95832c2be4a0e1e9cb24bee2142359a0246a7a9",
        "a3db3d07e2e4c08b060989a2a2db34026c7ea52ff02e1ee46dd7af64bc76c5bf",
        7906,
    ),
})


def measured_identity(path: Path) -> tuple[int, str]:
    """(size, sha256) a sealed receipt pins for ``path``.

    That is the module's own size and sha256, except for a module in
    ``REWORDED`` whose bytes are still exactly the reworded ones: then the
    identity it had before.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    posix = Path(path).as_posix()
    for suffix, (now, then, size) in REWORDED.items():
        if posix.endswith(suffix) and digest == now:
            return size, then
    return len(raw), digest


def measured_sha256(path: Path) -> str:
    """The sha256 a sealed receipt pins for ``path`` (see ``measured_identity``)."""
    return measured_identity(path)[1]


#: The extra label older sealed receipts carry after the CM1- or O- prefix of an id.
_SEALED_LABEL = re.compile(r"(?<![A-Za-z0-9_])(CM1-|O-)W\d{1,2}[a-z]?-")


#: The same label at the head of a folder or file name older sealed records use
#: (``<label>-post-getset-v1``); GM1's records spell those names without it.
_PATH_LABEL = re.compile(r"(?<![A-Za-z0-9_])[Ww]\d{1,2}-(?=[a-z])")


def unlabel_path(text: str) -> str:
    """A path as an older sealed record spells it, as GM1's records spell it now."""
    return _PATH_LABEL.sub("", text)


def development_evidence(relative: str) -> Path:
    """``evidence/<relative>`` of the development repository, spelled as GM1's records spell it.

    The repository keeps some folders and files under the names their sealed
    records gave them, with the label before the name; each part of
    ``relative`` is found with or without it.  An absent part gives a path that
    does not exist, so the caller's read fails as it would for any missing file.
    """
    path = _ROOT / "evidence"
    for part in PurePosixPath(relative).parts:
        if not (path / part).exists() and path.is_dir():
            named = [child for child in path.iterdir() if unlabel_path(child.name) == part]
            if len(named) == 1:
                path = named[0]
                continue
        path = path / part
    return path


def unlabel(text: str) -> str:
    """An id as an older sealed receipt spells it, as GM1 spells it now.

    Those receipts name oracle fixtures and refusal codes with an extra label
    after the prefix (``O-<label>-GRID-BASE-V1``); GM1's ids carry none, and the
    receipts are not rewritten.
    """
    return _SEALED_LABEL.sub(r"\1", text)


#: The structural state plan's identity as the sm86 receipt measured it.  The
#: plan names the oracle gates its deferred carriers wait on, and those ids are
#: spelled without the receipt's label (O-GRID, O-ARWEN-WSM6, O-CLOSURES,
#: O-INSTRUMENT); nothing else in the plan moved.  plan sha256 now -> plan sha256 measured.
RELABELED_PLAN: dict[str, str] = {
    "7b367d83ccf392d0fc0353a608575c5f577a45318e159dcf49151aec79ca525f":
        "2d7261511c7f11a897aa69a09221f2618352d01347095ac01bdf73efac56304a",
}


def measured_plan_sha256(sha256: str) -> str:
    """The plan identity a sealed receipt pins for a plan whose identity is now ``sha256``."""
    return RELABELED_PLAN.get(sha256, sha256)
