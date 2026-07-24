#!/usr/bin/env python3
"""Export a qualified frozen Theory 3.3 artifact to a Slang module."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def literal(value: float) -> str:
    return f"{float(value):.17g}"


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def effective_lattice(
    state: dict[str, Any],
    prefix: str,
) -> list[float]:
    anchor = float(state[f"{prefix}.anchor"])
    maximum_drop = float(state[f"{prefix}.maximum_drop"])
    values = [anchor]
    for raw in state[f"{prefix}.raw_decrements"]:
        values.append(values[-1] - maximum_drop * sigmoid(float(raw)))
    return values


def array(name: str, values: list[float]) -> str:
    body = ", ".join(literal(value) for value in values)
    return (
        f"static const int {name}_COUNT = {len(values)};\n"
        f"static const float {name}[{len(values)}] = {{ {body} }};"
    )


def render(artifact: dict[str, Any], base_source: str) -> str:
    variants = artifact["variants"]
    config = artifact["data_config"]
    controller_state = variants["controller"]["state"]
    negative_values = effective_lattice(controller_state, "negative")
    positive_values = effective_lattice(controller_state, "positive")
    drift_state = variants["drift_master"]["state"]
    drift_weights = [
        softplus(float(value)) for value in drift_state["raw_weights"]
    ]
    drift_quadratic = softplus(float(drift_state["raw_quadratic"]))
    drift_knots = [float(value) for value in drift_state["knots"]]
    drift_sharpness = float(drift_state["sharpness"])
    energy_scale = float(variants["drift_master"]["energy_scale"])
    drift_scale = float(variants["drift_master"]["drift_scale"])
    phase = variants["phase"]
    mobility = variants["mobility"]["state"]
    input_weight = mobility["input_layer.weight"]
    input_bias = mobility["input_layer.bias"]
    output_weight = mobility["output_layer.weight"]
    output_bias = mobility["output_layer.bias"]
    width = int(variants["mobility"]["width"])

    hidden_lines = []
    for row in range(width):
        terms = " + ".join(
            f"{literal(input_weight[row][column])} * context{column}"
            for column in range(3)
        )
        hidden_lines.append(
            f"    float h{row} = tanh({literal(input_bias[row])}"
            f" + {terms});"
        )
    output_lines = []
    for row in range(3):
        terms = " + ".join(
            f"{literal(output_weight[row][column])} * h{column}"
            for column in range(width)
        )
        output_lines.append(
            f"    float raw{row} = {literal(output_bias[row])}"
            f" + {terms};"
        )

    drift_terms = []
    for knot, weight in zip(drift_knots, drift_weights, strict=True):
        origin = -drift_sharpness * knot
        derivative_origin = sigmoid(origin) * drift_sharpness
        drift_terms.append(
            "        + "
            f"{literal(weight)} * (qualifiedSoftplus("
            f"{literal(drift_sharpness)} * (y - {literal(knot)}))"
            f" - {literal(softplus(origin))}"
            f" - {literal(derivative_origin)} * y)"
        )
    drift_expression = "\n".join(drift_terms)
    suffix = f"""

// Generated from theory33_frozen_variants.json. Do not hand edit.
static const float T33Q_ENERGY_SCALE = {literal(energy_scale)};
static const float T33Q_DRIFT_SCALE = {literal(drift_scale)};
static const float T33Q_PHASE_BASE = {literal(phase["base"])};
static const float T33Q_PHASE_T = {literal(phase["temperature_coefficient"])};
static const float T33Q_PHASE_N = {literal(phase["density_coefficient"])};
static const float T33Q_PHASE_F = {literal(phase["fraction_coefficient"])};
static const float T33Q_DENSITY_MIN = {literal(config["density_minimum"])};
static const float T33Q_DENSITY_MAX = {literal(config["density_maximum"])};
static const float T33Q_TEMPERATURE_MIN =
    {literal(config["temperature_minimum"])};
static const float T33Q_TEMPERATURE_MAX =
    {literal(config["temperature_maximum"])};
static const float T33Q_FRACTION_MIN = {literal(config["fraction_minimum"])};
static const float T33Q_FRACTION_MAX = {literal(config["fraction_maximum"])};

{array("T33Q_NEGATIVE_LATTICE", negative_values)}
{array("T33Q_POSITIVE_LATTICE", positive_values)}

[Differentiable]
float qualifiedSoftplus(float value)
{{
    return max(value, 0.0) + log(1.0 + exp(-abs(value)));
}}

[Differentiable]
float qualifiedSigmoid(float value)
{{
    return 1.0 / (1.0 + exp(-value));
}}

[Differentiable]
float qualifiedDriftNetwork(float y)
{{
    return
        0.5 * {literal(drift_quadratic)} * y * y
{drift_expression};
}}

[Differentiable]
float qualifiedPhaseThreshold(
    float density,
    float carrier,
    float entropy
)
{{
    float pressure =
        exp((T33_GAMMA_AD - 1.0) * entropy)
        * pow(density, T33_GAMMA_AD);
    float temperature = pressure / max(density, T33_EPSILON);
    float densityCoordinate = clamp(
        2.0
        * (density - T33Q_DENSITY_MIN)
        / (T33Q_DENSITY_MAX - T33Q_DENSITY_MIN)
        - 1.0,
        -1.5,
        1.5
    );
    float temperatureCoordinate = clamp(
        2.0
        * (log(max(temperature, T33_EPSILON)) - log(T33Q_TEMPERATURE_MIN))
        / (log(T33Q_TEMPERATURE_MAX) - log(T33Q_TEMPERATURE_MIN))
        - 1.0,
        -1.5,
        1.5
    );
    float fractionCoordinate = clamp(
        2.0
        * (carrier / max(density, T33_EPSILON) - T33Q_FRACTION_MIN)
        / (T33Q_FRACTION_MAX - T33Q_FRACTION_MIN)
        - 1.0,
        -1.5,
        1.5
    );
    return
        T33Q_PHASE_BASE
        + T33Q_PHASE_T * temperatureCoordinate
        + T33Q_PHASE_N * densityCoordinate
        + T33Q_PHASE_F * fractionCoordinate;
}}

[Differentiable]
float qualifiedTheory33LambdaValue(
    float nSquared,
    float dSquared,
    float xSquared,
    float entropy
)
{{
    float base = theory33LambdaValue(
        nSquared, dSquared, xSquared, entropy
    );
    float density = sqrt(max(nSquared, T33_EPSILON));
    float carrier = sqrt(max(dSquared, T33_EPSILON));
    float product = max(density * carrier, T33_EPSILON);
    float drift = max(xSquared / product - 1.0, 0.0);
    float threshold = qualifiedPhaseThreshold(
        density, carrier, entropy
    );
    if (drift < threshold)
        return base;
    float y = drift / T33Q_DRIFT_SCALE;
    float thresholdY = threshold / T33Q_DRIFT_SCALE;
    return
        base
        - T33Q_ENERGY_SCALE
        * (qualifiedDriftNetwork(y) - qualifiedDriftNetwork(thresholdY));
}}

[Differentiable]
void qualifiedTheory33Master(
    int sample,
    IDiffTensor<float, 2> features,
    IWDiffTensor<float, 1> output
)
{{
    output[sample] = qualifiedTheory33LambdaValue(
        features[sample, 0],
        features[sample, 1],
        features[sample, 2],
        features[sample, 3]
    );
}}

float qualifiedLattice(
    float coordinate,
    const float values[T33Q_NEGATIVE_LATTICE_COUNT]
)
{{
    float scaled =
        (clamp(coordinate, -1.0, 1.0) + 1.0)
        * float(T33Q_NEGATIVE_LATTICE_COUNT - 1)
        / 2.0;
    int index = clamp(
        int(floor(scaled)),
        0,
        T33Q_NEGATIVE_LATTICE_COUNT - 2
    );
    float fraction = scaled - float(index);
    return lerp(values[index], values[index + 1], fraction);
}}

float qualifiedKappaResidual(float z)
{{
    float negativeCoordinate = clamp(
        2.0 * (z - ({literal(-1.0)}))
        / ({literal(-2.0 / 3.0 - 0.05)} - ({literal(-1.0)}))
        - 1.0,
        -1.0,
        1.0
    );
    float positiveCoordinate = clamp(
        2.0 * (z - {literal(2.0 / 3.0 - 0.05)})
        / ({literal(1.05)} - {literal(2.0 / 3.0 - 0.05)})
        - 1.0,
        -1.0,
        1.0
    );
    return z < 0.0
        ? qualifiedLattice(negativeCoordinate, T33Q_NEGATIVE_LATTICE)
        : qualifiedLattice(positiveCoordinate, T33Q_POSITIVE_LATTICE);
}}

void qualifiedKappa(
    int sample,
    ITensor<float, 1> z,
    IWTensor<float, 1> output
)
{{
    output[sample] = qualifiedKappaResidual(z[sample]);
}}

void qualifiedMobilityValues(
    float context0,
    float context1,
    float context2,
    out float l00,
    out float l01,
    out float l11
)
{{
{chr(10).join(hidden_lines)}
{chr(10).join(output_lines)}
    float diagonal0 = qualifiedSoftplus(raw0) + 1.0e-6;
    float offDiagonal = raw1;
    float diagonal1 = qualifiedSoftplus(raw2) + 1.0e-6;
    l00 = diagonal0 * diagonal0;
    l01 = diagonal0 * offDiagonal;
    l11 = offDiagonal * offDiagonal + diagonal1 * diagonal1;
}}

void qualifiedMobility(
    int sample,
    ITensor<float, 2> inputContext,
    ITensor<float, 2> force,
    IWTensor<float, 2> output
)
{{
    float l00;
    float l01;
    float l11;
    qualifiedMobilityValues(
        inputContext[sample, 0],
        inputContext[sample, 1],
        inputContext[sample, 2],
        l00,
        l01,
        l11
    );
    float force0 = force[sample, 0];
    float force1 = force[sample, 1];
    float response0 = l00 * force0 + l01 * force1;
    float response1 = l01 * force0 + l11 * force1;
    output[sample, 0] = l00;
    output[sample, 1] = l01;
    output[sample, 2] = l11;
    output[sample, 3] = response0;
    output[sample, 4] = response1;
    output[sample, 5] =
        force0 * response0 + force1 * response1;
}}

void qualifiedCovariantMobility(
    int sample,
    ITensor<float, 2> inputContext,
    ITensor<float, 2> metric,
    ITensor<float, 2> baryonVelocity,
    ITensor<float, 2> carrierVelocity,
    ITensor<float, 2> force,
    IWTensor<float, 2> output
)
{{
    float l00;
    float l01;
    float l11;
    qualifiedMobilityValues(
        inputContext[sample, 0],
        inputContext[sample, 1],
        inputContext[sample, 2],
        l00,
        l01,
        l11
    );
    float4x4 g = float4x4(
        float4(
            metric[sample, 0],
            metric[sample, 1],
            metric[sample, 2],
            metric[sample, 3]
        ),
        float4(
            metric[sample, 4],
            metric[sample, 5],
            metric[sample, 6],
            metric[sample, 7]
        ),
        float4(
            metric[sample, 8],
            metric[sample, 9],
            metric[sample, 10],
            metric[sample, 11]
        ),
        float4(
            metric[sample, 12],
            metric[sample, 13],
            metric[sample, 14],
            metric[sample, 15]
        )
    );
    float4 u = float4(
        baryonVelocity[sample, 0],
        baryonVelocity[sample, 1],
        baryonVelocity[sample, 2],
        baryonVelocity[sample, 3]
    );
    float4 carrierU = float4(
        carrierVelocity[sample, 0],
        carrierVelocity[sample, 1],
        carrierVelocity[sample, 2],
        carrierVelocity[sample, 3]
    );
    float relativeGamma = -dot(u, mul(g, carrierU));
    float4 spatial = carrierU - relativeGamma * u;
    float spatialNorm = sqrt(max(dot(spatial, mul(g, spatial)), 0.0));
    if (spatialNorm > 1.0e-9)
    {{
        spatial /= spatialNorm;
    }}
    else
    {{
        float4 axis = float4(0.0, 1.0, 0.0, 0.0);
        float4 axisCovector = mul(g, axis);
        spatial = axis + u * dot(u, axisCovector);
        spatial /= sqrt(max(dot(spatial, mul(g, spatial)), 1.0e-18));
    }}
    float4 spatialCovector = mul(g, spatial);
    float4 projectedSpatial =
        spatial + u * dot(u, spatialCovector);
    float force0 = force[sample, 0];
    float force1 = force[sample, 1];
    float response0 = l00 * force0 + l01 * force1;
    float response1 = l01 * force0 + l11 * force1;
    float4 flux0 = -response0 * projectedSpatial;
    float4 flux1 = -response1 * projectedSpatial;
    output[sample, 0] = flux0.x;
    output[sample, 1] = flux0.y;
    output[sample, 2] = flux0.z;
    output[sample, 3] = flux0.w;
    output[sample, 4] = flux1.x;
    output[sample, 5] = flux1.y;
    output[sample, 6] = flux1.z;
    output[sample, 7] = flux1.w;
    output[sample, 8] =
        force0 * response0 + force1 * response1;
    output[sample, 9] = l00;
    output[sample, 10] = l01;
    output[sample, 11] = l11;
}}
"""
    return base_source.rstrip() + "\n" + suffix.lstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("theory33_frozen_variants.json"),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("theory33_hybrid.slang"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("theory33_qualified_frozen.slang"),
    )
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if not bool(
        artifact["metrics"]["continuation"]["final_audit"]["qualified"]
    ):
        raise RuntimeError("refusing to export an unqualified artifact")
    source = render(
        artifact,
        args.base.read_text(encoding="utf-8"),
    )
    args.output.write_text(source, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
