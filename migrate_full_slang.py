#!/usr/bin/env python3
"""Mechanical first-pass migration of the production GLSL equations to Slang.

The generated module deliberately keeps the GLSL implementation as the source
of truth while the differentiable entry points and hard-boundary annotations
are maintained in the Slang suffix/preamble below.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import json


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "carter_tesseract_kkt.comp"
OUTPUT = HERE / "carter_tesseract_full.slang"

PREAMBLE = r'''import "slangpy";

static const int DIM = 4;
static const int KKT_N = 4;
static const int KKT_STATES = 81;
static const float SAFE_EPS = 1.0e-12;
static const int MASTER_WIDTH = 32;
static const int MASTER_WEIGHT_COUNT = 1287;
static const int MW_SKIP_W = 0;
static const int MW_SKIP_B = 5;
static const int MW_L1_W = 6;
static const int MW_L1_B = 166;
static const int MW_L2_W = 198;
static const int MW_L2_B = 1222;
static const int MW_L3_W = 1254;
static const int MW_L3_B = 1286;

static const int H_J_PLUS = 0x0;
static const int H_J_MINUS = 0x1;
static const int H_G = 0x2;
static const int H_GAMMA = 0x3;
static const int H_Q_LIMITED = 0x4;
static const int H_THETA_STAR = 0x5;
static const int H_LAMBDA = 0x6;
static const int H_ENERGY = 0x7;
static const int H_J_LIMITED = 0x8;
static const int H_RESIDUAL = 0x9;
static const int H_X_PLUS = 0xA;
static const int H_X_MINUS = 0xB;
static const int H_X_CROSS = 0xC;
static const int H_KKT = 0xD;
static const int H_CLOSURE = 0xE;
static const int H_M = 0xF;

struct CarterParams
{
    float c;
    float kappa;
    float alpha;
    float chi;
    float zeta;
    float omegaDelta;
    float epsilonN;
    float crossingTolerance;
    float omegaE;
    float omegaT;
    float omegaPi;
    float omegaA;
    float lambda0;
    float lambdaP;
    float lambdaM;
    float lambdaC;
    float lambdaPP;
    float lambdaMM;
    float lambdaCC;
    float lambdaPM;
    float lambdaGamma;
    float lambdaQ;
    float lambdaGammaQ;
    float branchSplit;
    float4x4 targetT;
    uint elementCount;
    uint useFullW;
    int forceSigma;
    uint useNeuralLambda;
};

__MODEL_CONSTANTS__
'''

SUFFIX = r'''

// -----------------------------------------------------------------------------
// Compact differentiable recurrence.
//
// The complete diagnostic TesseractState above is retained for the exact primal
// implementation and for producing the hard active-set/classifier tape.  Slang's
// SPIR-V reverse emitter currently cannot lower the derivative of that very large
// aggregate.  The compact state below contains exactly the continuous quantities
// consumed by the recurrence.  KKT selection remains an exact primal operation
// and its selected value/tangent pair is detached, which is the piecewise-smooth
// derivative of the physical step away from switching surfaces.
// -----------------------------------------------------------------------------

struct CompactPhysicalState : IDifferentiable
{
    D1 jLimited;
    D1 residual;
    D1 M;
};

[Differentiable]
CompactPhysicalState closeCompactPhysical(
    D1 z,
    int sigma,
    D1 qLimited,
    DVec4 thetaStar,
    LambdaResult lambda
)
{
    DVec4 JPlus = currentField(z, H_J_PLUS, sigma, +1.0);
    DVec4 JMinus = currentField(z, H_J_MINUS, sigma, -1.0);
    DMat4 G = metricField(z, sigma);
    D1 Gamma = gammaField(z, sigma);

    D1 XPlus = invariantPlus(G, JPlus);
    D1 XMinus = invariantMinus(G, JMinus);
    D1 XCross = invariantCross(G, JPlus, JMinus);

    D1 BPlus = dscale(-2.0, lambda.dXPlus);
    D1 BMinus = dscale(-2.0, lambda.dXMinus);
    D1 entrainmentA = dscale(-1.0, lambda.dXCross);

    DVec4 JPlusCov = dmvec(G, JPlus);
    DVec4 JMinusCov = dmvec(G, JMinus);
    DVec4 piPlus = dvadd(
        dvscale(BPlus, JPlusCov),
        dvscale(entrainmentA, JMinusCov)
    );
    DVec4 piMinus = dvadd(
        dvscale(BMinus, JMinusCov),
        dvscale(entrainmentA, JPlusCov)
    );

    D1 Psi = lambda.value;
    Psi = dsub(Psi, dvdot(JPlus, piPlus));
    Psi = dsub(Psi, dvdot(JMinus, piMinus));
    DMat4 T = stressTensor(Psi, JPlus, JMinus, piPlus, piMinus);
    D1 energy = energyField(XPlus, XMinus, XCross, Gamma, thetaStar);

    CompactPhysicalState state;
    state.jLimited = limitedJField(z, qLimited, thetaStar);
    state.residual = residualField(Gamma, qLimited, thetaStar, XCross);
    state.M = freeEnergyField(
        energy, T, piPlus, piMinus, G, entrainmentA, XCross
    );
    return state;
}

[Differentiable]
CompactPhysicalState evaluateCompactPhysical(
    D1 z,
    int sigma,
    D1 qLimited,
    DVec4 thetaStar,
    IDiffTensor<float, 1> weights
)
{
    DVec4 JPlus = currentField(z, H_J_PLUS, sigma, +1.0);
    DVec4 JMinus = currentField(z, H_J_MINUS, sigma, -1.0);
    DMat4 G = metricField(z, sigma);
    D1 Gamma = gammaField(z, sigma);
    D1 XPlus = invariantPlus(G, JPlus);
    D1 XMinus = invariantMinus(G, JMinus);
    D1 XCross = invariantCross(G, JPlus, JMinus);
    LambdaResult lambda = lambdaField(
        XPlus, XMinus, XCross, Gamma, qLimited, weights
    );
    return closeCompactPhysical(z, sigma, qLimited, thetaStar, lambda);
}

[Differentiable]
float physicalStepContinuous(
    no_diff float z,
    no_diff float iota,
    no_diff int sigma,
    no_diff D1 activeQ,
    no_diff DVec4 activeTheta,
    no_diff D1 oppositeQ,
    no_diff DVec4 oppositeTheta,
    IDiffTensor<float, 1> weights
)
{
    CompactPhysicalState activeState =
        evaluateCompactPhysical(
            D(z, 1.0), sigma, activeQ, activeTheta, weights
        );
    CompactPhysicalState oppositeState =
        evaluateCompactPhysical(
            D(z, 1.0), -sigma, oppositeQ, oppositeTheta, weights
        );

    float MPlus = sigma > 0 ? activeState.M.v : oppositeState.M.v;
    float MMinus = sigma < 0 ? activeState.M.v : oppositeState.M.v;
    return recurrenceValue(
        z,
        activeState.jLimited.dz,
        activeState.residual.v,
        activeState.M.dz,
        MPlus - MMinus,
        iota
    );
}

[Differentiable]
void physicalStepKernel(
    int sample,
    ITensor<float, 1> zValues,
    ITensor<float, 2> hardTape,
    IDiffTensor<float, 1> weights,
    IWDiffTensor<float, 1> output
)
{
    D1 activeQ = D(hardTape[sample, 2], hardTape[sample, 3]);
    DVec4 activeTheta;
    activeTheta.v = float4(
        hardTape[sample, 4],
        hardTape[sample, 5],
        hardTape[sample, 6],
        0.0
    );
    activeTheta.dz = float4(
        hardTape[sample, 7],
        hardTape[sample, 8],
        hardTape[sample, 9],
        0.0
    );
    D1 oppositeQ = D(hardTape[sample, 10], hardTape[sample, 11]);
    DVec4 oppositeTheta;
    oppositeTheta.v = float4(
        hardTape[sample, 12],
        hardTape[sample, 13],
        hardTape[sample, 14],
        0.0
    );
    oppositeTheta.dz = float4(
        hardTape[sample, 15],
        hardTape[sample, 16],
        hardTape[sample, 17],
        0.0
    );
    output[sample] = physicalStepContinuous(
        zValues[sample],
        hardTape[sample, 0],
        int(hardTape[sample, 1]),
        activeQ,
        activeTheta,
        oppositeQ,
        oppositeTheta,
        weights
    );
}

// Stage one of the production differentiable path. Each (sample, side) lane
// evaluates one learned Carter master and materializes the constitutive response
// needed by the physical closure. side=0 is selected sigma; side=1 is -sigma.
[Differentiable]
void neuralClosureKernel(
    int index[2],
    ITensor<float, 1> zValues,
    ITensor<float, 2> hardTape,
    IDiffTensor<float, 1> weights,
    IWDiffTensor<float, 3> closure
)
{
    int sample = index[0];
    int side = index[1];
    float zValue = zValues[sample];
    int selected = int(hardTape[sample, 1]);
    int sigma = side == 0 ? selected : -selected;
    int base = side == 0 ? 2 : 10;
    D1 z = D(zValue, 1.0);
    D1 qLimited = D(
        hardTape[sample, base], hardTape[sample, base + 1]
    );

    DVec4 JPlus = currentField(z, H_J_PLUS, sigma, +1.0);
    DVec4 JMinus = currentField(z, H_J_MINUS, sigma, -1.0);
    DMat4 G = metricField(z, sigma);
    D1 Gamma = gammaField(z, sigma);
    D1 XPlus = invariantPlus(G, JPlus);
    D1 XMinus = invariantMinus(G, JMinus);
    D1 XCross = invariantCross(G, JPlus, JMinus);
    LambdaResult lambda = lambdaField(
        XPlus, XMinus, XCross, Gamma, qLimited, weights
    );

    closure[sample, side, 0] = lambda.value.v;
    closure[sample, side, 1] = lambda.value.dz;
    closure[sample, side, 2] = lambda.dXPlus.v;
    closure[sample, side, 3] = lambda.dXPlus.dz;
    closure[sample, side, 4] = lambda.dXMinus.v;
    closure[sample, side, 5] = lambda.dXMinus.dz;
    closure[sample, side, 6] = lambda.dXCross.v;
    closure[sample, side, 7] = lambda.dXCross.dz;
}

// A two-direction jet plus its mixed component.  If u and c,v are input
// directions, the components represent (f, D_u f, D_c f, D_v f, D_c D_v f).
// This is the smallest algebra that can contract all eight closure adjoints.
struct ClosureJet
{
    float p;
    float u;
    float c;
    float v;
    float cv;
};

ClosureJet closureJet(
    float p, float u, float c, float v, float cv
)
{
    ClosureJet result;
    result.p = p;
    result.u = u;
    result.c = c;
    result.v = v;
    result.cv = cv;
    return result;
}

ClosureJet jetZero()
{
    return closureJet(0.0, 0.0, 0.0, 0.0, 0.0);
}

ClosureJet jetAdd(ClosureJet a, ClosureJet b)
{
    return closureJet(
        a.p + b.p,
        a.u + b.u,
        a.c + b.c,
        a.v + b.v,
        a.cv + b.cv
    );
}

ClosureJet jetScale(float scale, ClosureJet value)
{
    return closureJet(
        scale * value.p,
        scale * value.u,
        scale * value.c,
        scale * value.v,
        scale * value.cv
    );
}

float jetDot(ClosureJet a, ClosureJet b)
{
    return
          a.p * b.p
        + a.u * b.u
        + a.c * b.c
        + a.v * b.v
        + a.cv * b.cv;
}

ClosureJet jetTanh(ClosureJet x)
{
    float y = tanh(x.p);
    float first = 1.0 - y * y;
    float second = -2.0 * y * first;
    return closureJet(
        y,
        first * x.u,
        first * x.c,
        first * x.v,
        first * x.cv + second * x.c * x.v
    );
}

ClosureJet jetTanhBackward(
    ClosureJet x, ClosureJet y, ClosureJet adjointY
)
{
    float first = 1.0 - y.p * y.p;
    float second = -2.0 * y.p * first;
    float third = first * (6.0 * y.p * y.p - 2.0);
    return closureJet(
          adjointY.p * first
        + adjointY.u * second * x.u
        + adjointY.c * second * x.c
        + adjointY.v * second * x.v
        + adjointY.cv * (
              second * x.cv + third * x.c * x.v
          ),
        adjointY.u * first,
        adjointY.c * first + adjointY.cv * second * x.v,
        adjointY.v * first + adjointY.cv * second * x.c,
        adjointY.cv * first
    );
}

// Exact manually lowered VJP of neuralClosureKernel. Slang's synthesized
// higher-order reverse is mathematically equivalent but takes many minutes to
// optimize on SPIR-V. This kernel compiles as ordinary shader code and performs
// the same mixed-direction reverse directly.
void neuralClosureBackwardKernel(
    int index[2],
    ITensor<float, 1> zValues,
    ITensor<float, 2> hardTape,
    ITensor<float, 1> weights,
    IWTensor<float, 3> sampleGradient,
    ITensor<float, 3> closureAdjoint
)
{
    int sample = index[0];
    int side = index[1];
    float zValue = zValues[sample];
    int selected = int(hardTape[sample, 1]);
    int sigma = side == 0 ? selected : -selected;
    int base = side == 0 ? 2 : 10;
    D1 z = D(zValue, 1.0);
    D1 qLimited = D(
        hardTape[sample, base], hardTape[sample, base + 1]
    );
    DVec4 JPlus = currentField(z, H_J_PLUS, sigma, +1.0);
    DVec4 JMinus = currentField(z, H_J_MINUS, sigma, -1.0);
    DMat4 G = metricField(z, sigma);
    D1 inputDual[5];
    inputDual[0] = invariantPlus(G, JPlus);
    inputDual[1] = invariantMinus(G, JMinus);
    inputDual[2] = invariantCross(G, JPlus, JMinus);
    inputDual[3] = gammaField(z, sigma);
    inputDual[4] = qLimited;

    float adjointValue = closureAdjoint[sample, side, 0];
    float adjointDz = closureAdjoint[sample, side, 1];
    ClosureJet inputJet[5];
    for (int i = 0; i < 5; ++i)
    {
        float gradientAdjoint = i < 3
            ? closureAdjoint[sample, side, 2 + 2 * i]
            : 0.0;
        float gradientDzAdjoint = i < 3
            ? closureAdjoint[sample, side, 3 + 2 * i]
            : 0.0;
        inputJet[i] = closureJet(
            inputDual[i].v,
            adjointDz * inputDual[i].dz + gradientAdjoint,
            gradientDzAdjoint,
            inputDual[i].dz,
            0.0
        );
    }

    ClosureJet pre1[MASTER_WIDTH] = {};
    ClosureJet hidden1[MASTER_WIDTH] = {};
    ClosureJet pre2[MASTER_WIDTH] = {};
    ClosureJet hidden2[MASTER_WIDTH] = {};
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        ClosureJet pre = closureJet(
            weights[MW_L1_B + neuron], 0.0, 0.0, 0.0, 0.0
        );
        for (int i = 0; i < 5; ++i)
        {
            pre = jetAdd(
                pre,
                jetScale(
                    weights[MW_L1_W + neuron * 5 + i],
                    inputJet[i]
                )
            );
        }
        pre1[neuron] = pre;
        hidden1[neuron] = jetTanh(pre);
    }
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        ClosureJet pre = closureJet(
            weights[MW_L2_B + neuron], 0.0, 0.0, 0.0, 0.0
        );
        for (int i = 0; i < MASTER_WIDTH; ++i)
        {
            pre = jetAdd(
                pre,
                jetScale(
                    weights[MW_L2_W + neuron * MASTER_WIDTH + i],
                    hidden1[i]
                )
            );
        }
        pre2[neuron] = pre;
        hidden2[neuron] = jetTanh(pre);
    }

    // The contracted objective is a*Lambda + D_u Lambda + D_c D_v Lambda.
    ClosureJet outputAdjoint = closureJet(
        adjointValue, 1.0, 0.0, 0.0, 1.0
    );
    sampleGradient[sample, side, MW_SKIP_B] = outputAdjoint.p;
    sampleGradient[sample, side, MW_L3_B] = outputAdjoint.p;
    for (int i = 0; i < 5; ++i)
    {
        sampleGradient[sample, side, MW_SKIP_W + i] =
            jetDot(outputAdjoint, inputJet[i]);
    }

    ClosureJet adjointHidden2[MASTER_WIDTH] = {};
    ClosureJet adjointHidden1[MASTER_WIDTH] = {};
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        sampleGradient[sample, side, MW_L3_W + neuron] =
            jetDot(outputAdjoint, hidden2[neuron]);
        adjointHidden2[neuron] = jetScale(
            weights[MW_L3_W + neuron], outputAdjoint
        );
    }

    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        ClosureJet adjointPre = jetTanhBackward(
            pre2[neuron], hidden2[neuron], adjointHidden2[neuron]
        );
        sampleGradient[sample, side, MW_L2_B + neuron] = adjointPre.p;
        for (int i = 0; i < MASTER_WIDTH; ++i)
        {
            int weightIndex =
                MW_L2_W + neuron * MASTER_WIDTH + i;
            sampleGradient[sample, side, weightIndex] =
                jetDot(adjointPre, hidden1[i]);
            adjointHidden1[i] = jetAdd(
                adjointHidden1[i],
                jetScale(weights[weightIndex], adjointPre)
            );
        }
    }

    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        ClosureJet adjointPre = jetTanhBackward(
            pre1[neuron], hidden1[neuron], adjointHidden1[neuron]
        );
        sampleGradient[sample, side, MW_L1_B + neuron] = adjointPre.p;
        for (int i = 0; i < 5; ++i)
        {
            sampleGradient[
                sample, side, MW_L1_W + neuron * 5 + i
            ] = jetDot(adjointPre, inputJet[i]);
        }
    }
}

void reduceWeightGradientKernel(
    int weightIndex,
    ITensor<float, 3> sampleGradient,
    IWTensor<float, 1> weightGradient
)
{
    float total = 0.0;
    [MaxIters(4096)]
    for (uint sample = 0; sample < sampleGradient.shape[0]; ++sample)
    {
        for (int side = 0; side < 2; ++side)
            total += sampleGradient[sample, side, weightIndex];
    }
    weightGradient[weightIndex] = total;
}

[Differentiable]
LambdaResult loadClosure(
    int sample,
    int side,
    IDiffTensor<float, 3> closure
)
{
    LambdaResult lambda;
    lambda.value = D(
        closure[sample, side, 0], closure[sample, side, 1]
    );
    lambda.dXPlus = D(
        closure[sample, side, 2], closure[sample, side, 3]
    );
    lambda.dXMinus = D(
        closure[sample, side, 4], closure[sample, side, 5]
    );
    lambda.dXCross = D(
        closure[sample, side, 6], closure[sample, side, 7]
    );
    return lambda;
}

// Stage two: exact continuous Carter physics using the frozen hard tape and the
// differentiable learned constitutive response produced by neuralClosureKernel.
[Differentiable]
void physicalClosureKernel(
    int sample,
    ITensor<float, 1> zValues,
    ITensor<float, 2> hardTape,
    IDiffTensor<float, 3> closure,
    IWDiffTensor<float, 1> output
)
{
    float z = zValues[sample];
    int sigma = int(hardTape[sample, 1]);
    D1 activeQ = D(hardTape[sample, 2], hardTape[sample, 3]);
    DVec4 activeTheta;
    activeTheta.v = float4(
        hardTape[sample, 4],
        hardTape[sample, 5],
        hardTape[sample, 6],
        0.0
    );
    activeTheta.dz = float4(
        hardTape[sample, 7],
        hardTape[sample, 8],
        hardTape[sample, 9],
        0.0
    );
    D1 oppositeQ = D(hardTape[sample, 10], hardTape[sample, 11]);
    DVec4 oppositeTheta;
    oppositeTheta.v = float4(
        hardTape[sample, 12],
        hardTape[sample, 13],
        hardTape[sample, 14],
        0.0
    );
    oppositeTheta.dz = float4(
        hardTape[sample, 15],
        hardTape[sample, 16],
        hardTape[sample, 17],
        0.0
    );

    CompactPhysicalState activeState = closeCompactPhysical(
        D(z, 1.0),
        sigma,
        activeQ,
        activeTheta,
        loadClosure(sample, 0, closure)
    );
    CompactPhysicalState oppositeState = closeCompactPhysical(
        D(z, 1.0),
        -sigma,
        oppositeQ,
        oppositeTheta,
        loadClosure(sample, 1, closure)
    );
    float MPlus = sigma > 0 ? activeState.M.v : oppositeState.M.v;
    float MMinus = sigma < 0 ? activeState.M.v : oppositeState.M.v;
    output[sample] = recurrenceValue(
        z,
        activeState.jLimited.dz,
        activeState.residual.v,
        activeState.M.dz,
        MPlus - MMinus,
        hardTape[sample, 0]
    );
}

// Exact hard tape layout:
//   0 iota, 1 sigma,
//   2..9 active q/theta values and d/dz tangents,
//   10..17 opposite q/theta values and d/dz tangents,
//   18 active mask, 19 classifier mask, 20 active KKT validity,
//   21 opposite active mask, 22 opposite KKT validity,
//   23..24 active/opposite KKT feasibility margins,
//   25..26 active/opposite classifier margins,
//   27..28 active/opposite selected-system condition estimates,
//   29 opposite classifier mask,
//   30 parity cell, 31 distance to the nearest parity boundary.
void fullPhysicalTapeKernel(
    int sample,
    ITensor<float, 1> zValues,
    IDiffTensor<float, 1> weights,
    IWTensor<float, 2> hardTape
)
{
    float z = zValues[sample];
    int sigma = selectedSigma(z);
    TesseractState activeStateT =
        evaluateTesseract(D(z, 1.0), sigma, weights);
    TesseractState oppositeStateT =
        evaluateTesseract(D(z, 1.0), -sigma, weights);
    float MPlus = sigma > 0
        ? activeStateT.scalar[H_M].v
        : oppositeStateT.scalar[H_M].v;
    float MMinus = sigma < 0
        ? activeStateT.scalar[H_M].v
        : oppositeStateT.scalar[H_M].v;
    float provisional = recurrenceValue(
        z,
        activeStateT.scalar[H_J_LIMITED].dz,
        activeStateT.scalar[H_RESIDUAL].v,
        activeStateT.scalar[H_M].dz,
        MPlus - MMinus,
        0.0
    );
    bool crossed = boundaryCrossed(
        z, provisional, sigma, activeStateT, weights
    );
    hardTape[sample, 0] = crossed ? 1.0 : 0.0;
    hardTape[sample, 1] = float(sigma);
    hardTape[sample, 2] = activeStateT.scalar[H_Q_LIMITED].v;
    hardTape[sample, 3] = activeStateT.scalar[H_Q_LIMITED].dz;
    hardTape[sample, 4] = activeStateT.vector[H_THETA_STAR].v.x;
    hardTape[sample, 5] = activeStateT.vector[H_THETA_STAR].v.y;
    hardTape[sample, 6] = activeStateT.vector[H_THETA_STAR].v.z;
    hardTape[sample, 7] = activeStateT.vector[H_THETA_STAR].dz.x;
    hardTape[sample, 8] = activeStateT.vector[H_THETA_STAR].dz.y;
    hardTape[sample, 9] = activeStateT.vector[H_THETA_STAR].dz.z;
    hardTape[sample, 10] = oppositeStateT.scalar[H_Q_LIMITED].v;
    hardTape[sample, 11] = oppositeStateT.scalar[H_Q_LIMITED].dz;
    hardTape[sample, 12] = oppositeStateT.vector[H_THETA_STAR].v.x;
    hardTape[sample, 13] = oppositeStateT.vector[H_THETA_STAR].v.y;
    hardTape[sample, 14] = oppositeStateT.vector[H_THETA_STAR].v.z;
    hardTape[sample, 15] = oppositeStateT.vector[H_THETA_STAR].dz.x;
    hardTape[sample, 16] = oppositeStateT.vector[H_THETA_STAR].dz.y;
    hardTape[sample, 17] = oppositeStateT.vector[H_THETA_STAR].dz.z;
    hardTape[sample, 18] = float(activeStateT.kkt.activeMask);
    hardTape[sample, 19] = float(activeStateT.classifierMask);
    hardTape[sample, 20] = activeStateT.kkt.valid != 0 ? 1.0 : 0.0;
    hardTape[sample, 21] = float(oppositeStateT.kkt.activeMask);
    hardTape[sample, 22] =
        oppositeStateT.kkt.valid != 0 ? 1.0 : 0.0;
    hardTape[sample, 23] = activeStateT.kkt.feasibilityMargin;
    hardTape[sample, 24] = oppositeStateT.kkt.feasibilityMargin;
    float activeClassifierMargin = 3.402823e38;
    float oppositeClassifierMargin = 3.402823e38;
    for (int i = 0; i < 4; ++i)
    {
        activeClassifierMargin = min(
            activeClassifierMargin,
            abs(activeStateT.classifierMargin[i])
        );
        oppositeClassifierMargin = min(
            oppositeClassifierMargin,
            abs(oppositeStateT.classifierMargin[i])
        );
    }
    hardTape[sample, 25] = activeClassifierMargin;
    hardTape[sample, 26] = oppositeClassifierMargin;
    hardTape[sample, 27] = activeStateT.kkt.conditionEstimate;
    hardTape[sample, 28] = oppositeStateT.kkt.conditionEstimate;
    hardTape[sample, 29] = float(oppositeStateT.classifierMask);
    float parityCoordinate = abs(1.5 * (z + PARAM_c));
    float parityCell = floor(parityCoordinate);
    hardTape[sample, 30] = parityCell;
    hardTape[sample, 31] = min(
        parityCoordinate - parityCell,
        parityCell + 1.0 - parityCoordinate
    );
}

// Visualization-only readback of the sixteen Carter/tesseract vertices.
// Channels are scalar value, scalar z-tangent, vector norm, and matrix norm.
void carterDiagnosticKernel(
    int sample,
    ITensor<float, 1> zValues,
    IDiffTensor<float, 1> weights,
    IWTensor<float, 3> diagnostics
)
{
    float z = zValues[sample];
    int sigma = selectedSigma(z);
    TesseractState state =
        evaluateTesseract(D(z, 1.0), sigma, weights);
    for (int vertex = 0; vertex < 16; ++vertex)
    {
        float matrixNormSquared = 0.0;
        for (int row = 0; row < 4; ++row)
        {
            float4 values = state.matrix4[vertex].v[row];
            matrixNormSquared += dot(values, values);
        }
        diagnostics[sample, vertex, 0] = state.scalar[vertex].v;
        diagnostics[sample, vertex, 1] = state.scalar[vertex].dz;
        diagnostics[sample, vertex, 2] = length(state.vector[vertex].v);
        diagnostics[sample, vertex, 3] = sqrt(matrixNormSquared);
    }
}

[Differentiable]
float4 fullPhysicalStep(no_diff float z, IDiffTensor<float, 1> weights)
{
    int sigma = no_diff(selectedSigma(z));
    TesseractState activeStateT = evaluateTesseract(D(z, 1.0), sigma, weights);
    TesseractState oppositeStateT = evaluateTesseract(D(z, 1.0), -sigma, weights);

    float dJLimited = activeStateT.scalar[H_J_LIMITED].dz;
    float residual = activeStateT.scalar[H_RESIDUAL].v;
    float dM = activeStateT.scalar[H_M].dz;
    float MPlus = sigma > 0
        ? activeStateT.scalar[H_M].v
        : oppositeStateT.scalar[H_M].v;
    float MMinus = sigma < 0
        ? activeStateT.scalar[H_M].v
        : oppositeStateT.scalar[H_M].v;
    float jumpM = MPlus - MMinus;
    float provisional = recurrenceValue(
        z, dJLimited, residual, dM, jumpM, 0.0
    );
    bool crossed = no_diff(
        boundaryCrossed(z, provisional, sigma, activeStateT, weights)
    );
    float result = recurrenceValue(
        z, dJLimited, residual, dM, jumpM, crossed ? 1.0 : 0.0
    );
    return float4(
        result,
        float(activeStateT.kkt.activeMask),
        float(activeStateT.classifierMask),
        activeStateT.kkt.valid != 0 ? 1.0 : 0.0
    );
}
'''


def generate(
    model_path: Path = HERE / "hybrid_model.json",
    output_path: Path = OUTPUT,
) -> None:
    source = SOURCE.read_text(encoding="utf-8")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    dynamics = model["dynamics"]
    scalar_values = {
        "c": dynamics["c"],
        "kappa": dynamics["kappa"],
        "alpha": dynamics["alpha"],
        "chi": 0.02,
        "zeta": dynamics["zeta"],
        "omegaDelta": dynamics["omega_delta"],
        "epsilonN": 1.0e-6,
        "crossingTolerance": 1.0e-5,
        "omegaE": 1.0,
        "omegaT": 0.1,
        "omegaPi": 0.1,
        "omegaA": 0.1,
        "lambda0": 0.5,
        "lambdaP": 0.2,
        "lambdaM": 0.2,
        "lambdaC": 0.1,
        "lambdaPP": 0.05,
        "lambdaMM": 0.05,
        "lambdaCC": 0.02,
        "lambdaPM": 0.01,
        "lambdaGamma": 0.1,
        "lambdaQ": 0.1,
        "lambdaGammaQ": 0.02,
        "branchSplit": 0.05,
    }
    constants = "\n".join(
        f"static const float PARAM_{name} = {float(value):.17g};"
        for name, value in scalar_values.items()
    )
    target = model["target_t"]
    # Generated code retains GLSL's targetT[B][A] indexing, so transpose.
    matrix_values = [
        float(target[row][column])
        for column in range(4)
        for row in range(4)
    ]
    constants += (
        "\nstatic const float4x4 TARGET_T = float4x4("
        + ", ".join(f"{value:.17g}" for value in matrix_values)
        + ");\n"
    )
    preamble = PREAMBLE.replace("__MODEL_CONSTANTS__", constants)
    start = source.index("struct D1")
    end = source.index("void main()")
    body = source[start:end]
    replacements = [
        (r"\bvec4\b", "float4"),
        (r"\bmat4\b", "float4x4"),
        (r"\buvec4\b", "uint4"),
        (r"\bmod\(", "fmod("),
    ]
    for pattern, replacement in replacements:
        body = re.sub(pattern, replacement, body)
    body = re.sub(
        r"(?m)^struct\s+([A-Za-z_]\w*)\s*\{",
        r"struct \1 : IDifferentiable\n{",
        body,
    )
    for name in scalar_values:
        body = body.replace(f"params.{name}", f"PARAM_{name}")
    body = body.replace("params.targetT", "TARGET_T")
    body = body.replace("params.useFullW", "0u")
    body = body.replace("params.forceSigma", "0")
    body = body.replace("params.useNeuralLambda", "1u")
    body = body.replace("masterWeights[", "weights[")
    body = body.replace("return W[widx(A, B, C, Dindex)];", "return 0.0;")
    body = body.replace(
        "state.kkt = solveBoundedKKT(qp, sigma);",
        "state.kkt = no_diff(solveBoundedKKT(qp, sigma));",
    )

    body = body.replace(
        "D1 qLimited\n)", "D1 qLimited,\n    IDiffTensor<float, 1> weights\n)"
    )
    body = body.replace(
        "neuralLambdaField(XPlus, XMinus, XCross, Gamma, qLimited)",
        "neuralLambdaField(XPlus, XMinus, XCross, Gamma, qLimited, weights)",
    )
    body = body.replace(
        "analyticLambdaField(XPlus, XMinus, XCross, Gamma, qLimited)",
        "analyticLambdaField(XPlus, XMinus, XCross, Gamma, qLimited, weights)",
    )
    body = body.replace(
        "TesseractState evaluateTesseract(D1 z, int sigma)",
        "TesseractState evaluateTesseract("
        "D1 z, int sigma, IDiffTensor<float, 1> weights)",
    )
    body = body.replace(
        "state.scalar[H_Q_LIMITED]\n    );",
        "state.scalar[H_Q_LIMITED],\n        weights\n    );",
    )
    body = body.replace(
        "TesseractState state0\n)",
        "TesseractState state0,\n    IDiffTensor<float, 1> weights\n)",
    )
    body = body.replace(
        "evaluateTesseract(D(midpoint, 1.0), sigma)",
        "evaluateTesseract(D(midpoint, 1.0), sigma, weights)",
    )
    body = body.replace(
        "evaluateTesseract(D(z1, 1.0), sigma)",
        "evaluateTesseract(D(z1, 1.0), sigma, weights)",
    )
    body = body.replace("A.v * x.v", "mul(A.v, x.v)")
    body = body.replace("A.dz * x.v", "mul(A.dz, x.v)")
    body = body.replace("A.v * x.dz", "mul(A.v, x.dz)")
    body = body.replace(
        "float parity = fmod(floor(abs(u)), 2.0);",
        "float parity = no_diff(fmod(floor(abs(u)), 2.0));",
    )
    for array_name in (
        "hidden1", "hidden2", "adjointHidden1", "adjointPre1",
        "adjointHidden2", "adjointPre2", "adjointInput",
    ):
        body = body.replace(
            f"D1 {array_name}[MASTER_WIDTH];",
            f"D1 {array_name}[MASTER_WIDTH] = {{}};",
        )
    def initialize_function_locals(
        text: str, start_marker: str, end_marker: str, declarations: tuple[str, ...]
    ) -> str:
        start_offset = text.index(start_marker)
        end_offset = text.index(end_marker, start_offset)
        function_text = text[start_offset:end_offset]
        for declaration in declarations:
            function_text = function_text.replace(
                declaration + ";", declaration + " = {};"
            )
        return text[:start_offset] + function_text + text[end_offset:]

    body = initialize_function_locals(
        body,
        "bool solveLinear(",
        "float matrixConditionEstimate(",
        (
            "float A[KKT_N * KKT_N]",
            "float b[KKT_N]",
        ),
    )
    body = initialize_function_locals(
        body,
        "float matrixConditionEstimate(",
        "int integerPower3(",
        (
            "float inverse[KKT_N * KKT_N]",
            "float rhs[KKT_N]",
            "float solution[KKT_N]",
        ),
    )
    body = initialize_function_locals(
        body,
        "DualBoxQP buildKKTProblem(",
        "KKTResult solveBoundedKKT(",
        (
            "DualBoxQP qp",
            "D1 Hd[KKT_N * KKT_N]",
            "D1 target[KKT_N]",
            "D1 lo[KKT_N]",
            "D1 hi[KKT_N]",
        ),
    )
    body = initialize_function_locals(
        body,
        "KKTResult solveBoundedKKT(",
        "D1 kktVariable(",
        (
            "KKTResult best",
            "float A[KKT_N * KKT_N]",
            "float dA[KKT_N * KKT_N]",
            "float b[KKT_N]",
            "float db[KKT_N]",
            "int states[KKT_N]",
            "float x[KKT_N]",
            "float gradient[KKT_N]",
            "float derivativeRhs[KKT_N]",
            "float dx[KKT_N]",
        ),
    )
    body = body.replace(
        "out float x[KKT_N]\n)\n{\n"
        "    float A[KKT_N * KKT_N] = {};",
        "out float x[KKT_N]\n)\n{\n"
        "    for (int i = 0; i < KKT_N; ++i)\n"
        "        x[i] = 0.0;\n\n"
        "    float A[KKT_N * KKT_N] = {};",
        1,
    )
    differentiable_returns = (
        "D1|DVec4|DMat4|DualBoxQP|KKTResult|LambdaResult|"
        "TesseractState|float|float4|bool"
    )
    body = re.sub(
        rf"(?m)^({differentiable_returns})\s+([A-Za-z_]\w*)\s*\(",
        r"[Differentiable]\n\1 \2(",
        body,
    )
    def annotate_loop(match: re.Match[str]) -> str:
        indentation = match.group(1)
        header = match.group(2)
        if "< KKT_STATES" in header:
            limit = 81
        elif "< KKT_N * KKT_N" in header:
            limit = 16
        elif "column + 1" in header:
            limit = 3
        elif "< KKT_N" in header:
            limit = 4
        elif "< MASTER_WIDTH" in header:
            limit = 32
        elif "< DIM" in header or "< 4" in header:
            limit = 4
        elif "< 16" in header:
            limit = 16
        elif "< 5" in header:
            limit = 5
        elif "< 3" in header:
            limit = 3
        elif "< exponent" in header:
            limit = 4
        else:
            raise RuntimeError(f"Unbounded generated loop: {header}")
        return (
            f"{indentation}[MaxIters({limit})]\n"
            f"{indentation}for ({header})"
        )

    body = re.sub(
        r"(?m)^(\s*)for \(([^)\n]+)\)",
        annotate_loop,
        body,
    )
    output_path.write_text(
        preamble + "\n" + body + SUFFIX, encoding="utf-8"
    )
    print(f"generated {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=HERE / "hybrid_model.json"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.model, arguments.output)
