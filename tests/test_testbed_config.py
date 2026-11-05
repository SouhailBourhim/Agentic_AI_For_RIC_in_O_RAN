"""M06: the three slices are configured the same way in every file of the testbed.

CI cannot start the testbed, but the most likely mistake can be caught without it: one file out
of step with the others (a slice missing in the gNB, a UE asking for a DNN the SMF does not
serve, a subscriber without its slice). The core rejects such a UE at attach time with a log
line that is easy to miss; here it is a failing test instead.

Values are read as text: PyYAML's default loader would turn `sd: 000001` into the integer 1, and
`000010` into the octal 8, while OAI reads it as a hex string.
"""

import re
from pathlib import Path

import pytest
import yaml

from verifier.objective import SLICES as OBJECTIVE_SLICES

TESTBED = Path(__file__).resolve().parent.parent / "testbed"
CORE_CONFIG = TESTBED / "oai-cn5g/conf/config.yaml"
DATABASE = TESTBED / "oai-cn5g/database/oai_db.sql"
GNB_CONFIG = TESTBED / "rfsim-flexric/conf/gnb.conf"
RAN_COMPOSE = TESTBED / "rfsim-flexric/docker-compose.yml"

# The slice design of M06 (docs/learn/M06-testbed.md): S-NSSAI (sst, sd), DNN, UE subnet, UE IMSI.
SLICES = {
    "embb": {"snssai": (1, 0x000001), "dnn": "embb", "subnet": "10.0.1.0/24", "imsi": "001010000000002"},
    "urllc": {"snssai": (2, 0x000001), "dnn": "urllc", "subnet": "10.0.2.0/24", "imsi": "001010000000003"},
    "be": {"snssai": (1, 0x000002), "dnn": "be", "subnet": "10.0.3.0/24", "imsi": "001010000000004"},
}
ALL_SNSSAIS = {s["snssai"] for s in SLICES.values()}


def snssai(entry: dict) -> tuple[int, int]:
    """(sst, sd) from a config.yaml entry; sd is a hex string, absent means 'no SD' (0xFFFFFF)."""
    return int(entry["sst"]), int(entry.get("sd", "FFFFFF"), 16)


@pytest.fixture(scope="module")
def core() -> dict:
    return yaml.load(CORE_CONFIG.read_text(), Loader=yaml.BaseLoader)  # every value as text


def test_three_distinct_slices() -> None:
    assert len(ALL_SNSSAIS) == 3


def test_amf_supports_exactly_the_three_slices(core) -> None:
    [plmn] = core["amf"]["plmn_support_list"]
    assert {snssai(n) for n in plmn["nssai"]} == ALL_SNSSAIS


def test_smf_upf_and_dnns_map_each_slice_to_its_dnn_and_subnet(core) -> None:
    expected = {s["snssai"]: s["dnn"] for s in SLICES.values()}
    smf = {snssai(e["sNssai"]): [d["dnn"] for d in e["dnnSmfInfoList"]] for e in core["smf"]["smf_info"]["sNssaiSmfInfoList"]}
    upf = {snssai(e["sNssai"]): [d["dnn"] for d in e["dnnUpfInfoList"]] for e in core["upf"]["upf_info"]["sNssaiUpfInfoList"]}
    local = {snssai(e["single_nssai"]): [e["dnn"]] for e in core["smf"]["local_subscription_infos"]}
    for table in (smf, upf, local):
        assert table == {k: [v] for k, v in expected.items()}
    assert {d["dnn"]: d["ipv4_subnet"] for d in core["dnns"]} == {s["dnn"]: s["subnet"] for s in SLICES.values()}


def test_gnb_announces_the_three_slices() -> None:
    found = {(int(sst), int(sd, 16)) for sst, sd in re.findall(r"sst\s*=\s*(\d+);\s*sd\s*=\s*0x([0-9a-fA-F]+);",
                                                               GNB_CONFIG.read_text())}
    assert found == ALL_SNSSAIS


@pytest.mark.parametrize("name", SLICES)
def test_each_ue_asks_for_its_slice(name: str) -> None:
    text = (TESTBED / f"rfsim-flexric/conf/nrue-{name}.conf").read_text()
    assert re.search(r'imsi\s*=\s*"(\d+)"', text).group(1) == SLICES[name]["imsi"]
    session = re.search(r'pdu_sessions\s*=\s*\(\{\s*dnn\s*=\s*"(\w+)";\s*nssai_sst\s*=\s*(\d+);\s*nssai_sd\s*=\s*0x([0-9a-fA-F]+);',
                        text)
    assert session is not None, f"nrue-{name}.conf: no pdu_sessions entry with dnn, nssai_sst and nssai_sd"
    dnn, sst, sd = session.groups()
    assert (dnn, (int(sst), int(sd, 16))) == (SLICES[name]["dnn"], SLICES[name]["snssai"])


@pytest.mark.parametrize("name", SLICES)
def test_compose_runs_each_ue_with_its_own_conf(name: str) -> None:
    services = yaml.load(RAN_COMPOSE.read_text(), Loader=yaml.BaseLoader)["services"]
    ue = services[f"oai-nr-ue-{name}"]
    assert ue["container_name"] == f"rfsim5g-oai-nr-ue-{name}"
    assert any(v.startswith(f"./conf/nrue-{name}.conf:") for v in ue["volumes"])
    assert "--uicc0.imsi" not in ue["environment"]["USE_ADDITIONAL_OPTIONS"]  # the IMSI comes from the conf


def test_each_subscriber_is_subscribed_to_its_slice() -> None:
    rows = re.findall(r"\('(\d{15})', '00101', '\{\\\"defaultSingleNssais\\\": \[\{\\\"sst\\\": (\d+), "
                      r"\\\"sd\\\": \\\"([0-9A-Fa-f]{6})\\\"\}\]\}'\)", DATABASE.read_text())
    assert {imsi: (int(sst), int(sd, 16)) for imsi, sst, sd in rows} == {s["imsi"]: s["snssai"] for s in SLICES.values()}


XAPP_SOURCE = Path(__file__).resolve().parent.parent / "xapps/kpm/xapp_kpm_slices.c"


def test_kpm_xapp_subscribes_to_exactly_the_three_slices() -> None:
    """M07a: the KPM xApp's subscription table (one per slice) must match the testbed slices,
    with the same names the KPI store uses."""
    table = re.search(r"static const slice_t slices\[\] = \{(.*?)\};", XAPP_SOURCE.read_text(), re.S).group(1)
    entries = re.findall(r'\{"(\w+)",\s*(\d+),\s*0x([0-9a-fA-F]{6})\}', table)
    assert {name: (int(sst), int(sd, 16)) for name, sst, sd in entries} == {
        name: s["snssai"] for name, s in SLICES.items()
    }


SLICE_PATCH = TESTBED / "gnb/slice-control.patch"
SLICE_XAPP = Path(__file__).resolve().parent.parent / "xapps/slice_ctrl/xapp_slice_ctrl.c"


def test_gnb_slice_scheduler_knows_the_three_slices_in_order() -> None:
    """M07b: slice id i in the E2 control = the i-th S-NSSAI in the gNB patch = SLICES[i]."""
    table = re.search(r"slice_nssai\[NR_SLICE_N\] = \{(.*?)\};", SLICE_PATCH.read_text(), re.S).group(1)
    entries = [(int(sst), int(sd, 16)) for sst, sd in re.findall(r"\{(\d+),\s*0x([0-9a-fA-F]{6})\}", table)]
    assert entries == [SLICES[s]["snssai"] for s in OBJECTIVE_SLICES]


def test_slice_control_xapp_sends_slices_in_the_same_order() -> None:
    """The xApp's line "a b c" is eMBB, URLLC, BE: ids 0, 1, 2, the backend's SLICES order."""
    table = re.search(r"labels\[\] = \{(.*?)\};", SLICE_XAPP.read_text(), re.S).group(1)
    assert tuple(re.findall(r'"(\w+)"', table)) == OBJECTIVE_SLICES


def test_backend_prb_count_is_the_gnb_carrier() -> None:
    from backends.flexric import N_PRB

    bandwidth = re.search(r"dl_carrierBandwidth\s*=\s*(\d+);", GNB_CONFIG.read_text()).group(1)
    assert N_PRB == int(bandwidth)


def test_prb_full_cell_is_the_downlink_share_of_the_tdd_pattern() -> None:
    """RRU.PrbTotDl averages over all slots; the backend divides by the downlink fraction of the
    TDD pattern written in gnb.conf (10 slots per 5 ms period at 30 kHz)."""
    from backends.flexric import PRB_FULL_CELL

    conf = GNB_CONFIG.read_text()
    value = lambda key: int(re.search(rf"{key}\s*=\s*(\d+);", conf).group(1))  # noqa: E731
    assert value("dl_UL_TransmissionPeriodicity") == 6  # 5 ms = 10 slots
    dl = value("nrofDownlinkSlots") + value("nrofDownlinkSymbols") / 14
    assert PRB_FULL_CELL == pytest.approx(dl / 10)
