from app.pipelines.graph.node_type1 import deterministic_type1_solver
from app.pipelines.graph.node_type2 import deterministic_physics_solver
from app.schemas import PredictRequest
from app.utils.json_utils import extract_json_object


def test_extract_json_object_salvages_partial_type1_payload():
    text = (
        '{"answer":"B","unit":"","explanation":"The codex is searchable online."'
        ',"premises_used":[0,1,2],"reasoning":{"type":"fol","steps":["x"]}'
    )
    data = extract_json_object(text)
    assert data["answer"] == "B"
    assert data["premises_used"] == [0, 1, 2]


def test_deterministic_solver_capacitance_prefers_microfarads():
    result = deterministic_physics_solver(
        "A capacitor stores a charge Q = 2.4 mC when connected to a voltage U = 16 V. "
        "Calculate the capacitance of the capacitor."
    )
    assert result is not None
    assert result["answer"] == "150"
    assert result["unit"] == "uF"


def test_deterministic_solver_resistance_keeps_ohm_mm2_units_consistent():
    result = deterministic_physics_solver(
        "A conductor has length l = 2 m, cross-sectional area S = 1 mm^2, and resistivity "
        "rho = 0.5 ohm*mm^2/m. Calculate its resistance."
    )
    assert result is not None
    assert result["answer"] == "1"
    assert result["unit"] == "ohm"


def test_deterministic_solver_formats_small_energy_in_scientific_notation():
    result = deterministic_physics_solver(
        "A capacitor has capacitance C = 47 uF and is connected to a potential difference "
        "U = 12 V. Calculate the energy stored in the capacitor."
    )
    assert result is not None
    assert result["answer"] == "3.38 × 10^-3"
    assert result["unit"] == "J"


def test_deterministic_type1_solver_handles_drone_chain_mcq():
    req = PredictRequest(
        query_id="REG_T1_0025",
        type="type1",
        query=(
            "Based on the drone delivery rules, which conclusion is logically supported?\n"
            "A. MedKit-7 cannot be dispatched because the route is blocked\n"
            "B. MedKit-7 has launch approval\n"
            "C. MedKit-7 is eligible to use the aerial corridor\n"
            "D. MedKit-7 is not a priority package"
        ),
        premises=[
            "If a package is medical and weighs under 2 kilograms, then it receives priority delivery status.",
            "If a package has priority delivery status and its route is clear, then it can be dispatched.",
            "If a package can be dispatched and the weather is safe, then it is eligible to use the aerial corridor.",
            "If a package is eligible to use the aerial corridor and an operator is assigned, then launch is approved.",
            "If an emergency waiver is approved and an alternate route is mapped, then the route is clear.",
            "The MedKit-7 package is medical.",
            "The MedKit-7 package weighs under 2 kilograms.",
            "An emergency waiver is approved for MedKit-7.",
            "An alternate route is mapped for MedKit-7.",
            "The weather is safe for MedKit-7.",
        ],
        options=["A", "B", "C", "D"],
    )
    result = deterministic_type1_solver(req)
    assert result is not None
    assert result["answer"] == "C"


def test_deterministic_type1_solver_handles_meta_establish_question():
    req = PredictRequest(
        query_id="REG_T1_0032",
        type="type1",
        query="Do the premises establish that the River Codex is safe for public release?",
        premises=[
            "If a manuscript is scanned at 600 dpi and its metadata is complete, then it is preservation-ready.",
            "If a manuscript is preservation-ready and its rights are cleared, then it is eligible for the public portal.",
            "If a manuscript is eligible for the public portal and OCR has been verified, then it is searchable online.",
            "If a manuscript contains personal data, then privacy review is required.",
            "If privacy review is required and redaction is complete, then the manuscript is safe for public release.",
            "The River Codex was scanned at 600 dpi.",
            "The River Codex metadata is complete.",
            "The River Codex rights are cleared.",
            "The River Codex OCR has been verified.",
            "The River Codex contains personal data.",
        ],
        options=["Yes", "No", "Uncertain"],
    )
    result = deterministic_type1_solver(req)
    assert result is not None
    assert result["answer"] == "No"


def test_deterministic_type1_solver_handles_greenhouse_chain_mcq():
    req = PredictRequest(
        query_id="REG_T1_0035",
        type="type1",
        query=(
            "Based on the smart greenhouse rules, which conclusion is supported?\n"
            "A. The reservoir lacks water\n"
            "B. Chemical treatment is allowed for Greenhouse Basil\n"
            "C. Irrigation is unnecessary because a heatwave is active\n"
            "D. Autonomous watering is allowed for Greenhouse Basil"
        ),
        premises=[
            "If soil moisture is low and a heatwave is active, then irrigation is needed.",
            "If irrigation is needed and the reservoir has water, then irrigation is scheduled.",
            "If irrigation is scheduled and sensor calibration is current, then autonomous watering is allowed.",
            "If pest risk is high, then pesticide review is required.",
            "If pesticide review is required and agronomist approval is given, then chemical treatment is allowed.",
            "Greenhouse Basil has low soil moisture.",
            "A heatwave is active for Greenhouse Basil.",
            "The Greenhouse Basil reservoir has water.",
            "The Greenhouse Basil sensor calibration is current.",
            "Greenhouse Basil has high pest risk.",
        ],
        options=["A", "B", "C", "D"],
    )
    result = deterministic_type1_solver(req)
    assert result is not None
    assert result["answer"] == "D"
