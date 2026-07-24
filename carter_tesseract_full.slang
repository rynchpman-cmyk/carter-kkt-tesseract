import "slangpy";

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

static const float PARAM_c = 0.24600024880977561;
static const float PARAM_kappa = 0.095999634176690857;
static const float PARAM_alpha = 0.054000458743839491;
static const float PARAM_chi = 0.02;
static const float PARAM_zeta = 0.014001046479233545;
static const float PARAM_omegaDelta = 0.49599874033503616;
static const float PARAM_epsilonN = 9.9999999999999995e-07;
static const float PARAM_crossingTolerance = 1.0000000000000001e-05;
static const float PARAM_omegaE = 1;
static const float PARAM_omegaT = 0.10000000000000001;
static const float PARAM_omegaPi = 0.10000000000000001;
static const float PARAM_omegaA = 0.10000000000000001;
static const float PARAM_lambda0 = 0.5;
static const float PARAM_lambdaP = 0.20000000000000001;
static const float PARAM_lambdaM = 0.20000000000000001;
static const float PARAM_lambdaC = 0.10000000000000001;
static const float PARAM_lambdaPP = 0.050000000000000003;
static const float PARAM_lambdaMM = 0.050000000000000003;
static const float PARAM_lambdaCC = 0.02;
static const float PARAM_lambdaPM = 0.01;
static const float PARAM_lambdaGamma = 0.10000000000000001;
static const float PARAM_lambdaQ = 0.10000000000000001;
static const float PARAM_lambdaGammaQ = 0.02;
static const float PARAM_branchSplit = 0.050000000000000003;
static const float4x4 TARGET_T = float4x4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);


struct D1 : IDifferentiable
{
    float v;
    float dz;
};

struct DVec4 : IDifferentiable
{
    float4 v;
    float4 dz;
};

struct DMat4 : IDifferentiable
{
    float4x4 v;
    float4x4 dz;
};

[Differentiable]
D1 D(float value, float derivative)
{
    D1 r;
    r.v = value;
    r.dz = derivative;
    return r;
}

[Differentiable]
D1 dadd(D1 a, D1 b)
{
    return D(a.v + b.v, a.dz + b.dz);
}

[Differentiable]
D1 dsub(D1 a, D1 b)
{
    return D(a.v - b.v, a.dz - b.dz);
}

[Differentiable]
D1 dneg(D1 a)
{
    return D(-a.v, -a.dz);
}

[Differentiable]
D1 dscale(float s, D1 a)
{
    return D(s * a.v, s * a.dz);
}

[Differentiable]
D1 dmul(D1 a, D1 b)
{
    return D(
        a.v * b.v,
        a.dz * b.v + a.v * b.dz
    );
}

[Differentiable]
D1 ddiv(D1 a, D1 b)
{
    float denominator = b.v * b.v;
    float guarded = max(abs(denominator), SAFE_EPS);
    float signedGuarded = denominator < 0.0 ? -guarded : guarded;

    return D(
        a.v / b.v,
        (a.dz * b.v - a.v * b.dz) / signedGuarded
    );
}

[Differentiable]
D1 dsquare(D1 a)
{
    return D(a.v * a.v, 2.0 * a.v * a.dz);
}

[Differentiable]
D1 dsin(D1 a)
{
    return D(sin(a.v), cos(a.v) * a.dz);
}

[Differentiable]
D1 dcos(D1 a)
{
    return D(cos(a.v), -sin(a.v) * a.dz);
}

[Differentiable]
D1 dtanh(D1 a)
{
    float t = tanh(a.v);
    return D(t, (1.0 - t * t) * a.dz);
}

[Differentiable]
D1 dsqrtSafe(D1 a)
{
    float value = sqrt(max(a.v, SAFE_EPS));
    return D(value, a.dz / max(2.0 * value, SAFE_EPS));
}

[Differentiable]
D1 dsmoothAbs(D1 a)
{
    return dsqrtSafe(dadd(dsquare(a), D(1.0e-8, 0.0)));
}

[Differentiable]
DVec4 dvzero()
{
    DVec4 r;
    r.v = float4(0.0);
    r.dz = float4(0.0);
    return r;
}

[Differentiable]
DMat4 dmzero()
{
    DMat4 r;
    r.v = float4x4(0.0);
    r.dz = float4x4(0.0);
    return r;
}

[Differentiable]
DVec4 dvadd(DVec4 a, DVec4 b)
{
    DVec4 r;
    r.v = a.v + b.v;
    r.dz = a.dz + b.dz;
    return r;
}

[Differentiable]
DVec4 dvsub(DVec4 a, DVec4 b)
{
    DVec4 r;
    r.v = a.v - b.v;
    r.dz = a.dz - b.dz;
    return r;
}

[Differentiable]
DVec4 dvscale(D1 a, DVec4 x)
{
    DVec4 r;
    r.v = a.v * x.v;
    r.dz = a.dz * x.v + a.v * x.dz;
    return r;
}

[Differentiable]
DVec4 dmvec(DMat4 A, DVec4 x)
{
    DVec4 r;
    r.v = mul(A.v, x.v);
    r.dz = mul(A.dz, x.v) + mul(A.v, x.dz);
    return r;
}

[Differentiable]
D1 dvdot(DVec4 a, DVec4 b)
{
    return D(
        dot(a.v, b.v),
        dot(a.dz, b.v) + dot(a.v, b.dz)
    );
}

[Differentiable]
D1 dvcomponent(DVec4 a, int component)
{
    return D(a.v[component], a.dz[component]);
}

void dmset(inout DMat4 A, int row, int column, D1 x)
{
    // GLSL matrices are indexed [column][row].
    A.v[column][row] = x.v;
    A.dz[column][row] = x.dz;
}

// -----------------------------------------------------------------------------
// Tesseract addressing.
// -----------------------------------------------------------------------------

[Differentiable]
float4 tesseractCoord(int vertex)
{
    return float4(
        (vertex & 0x8) != 0 ? 1.0 : -1.0,
        (vertex & 0x4) != 0 ? 1.0 : -1.0,
        (vertex & 0x2) != 0 ? 1.0 : -1.0,
        (vertex & 0x1) != 0 ? 1.0 : -1.0
    );
}

[Differentiable]
float vertexPhase(int vertex)
{
    float4 h = tesseractCoord(vertex);
    return dot(h, float4(0.31, 0.47, 0.73, 1.13));
}

// -----------------------------------------------------------------------------
// Explicit analytic fields at the primitive tesseract vertices.
// -----------------------------------------------------------------------------

[Differentiable]
DVec4 currentField(D1 z, int vertex, int sigma, float orientation)
{
    float4 h = tesseractCoord(vertex);
    float phase0 = vertexPhase(vertex);
    float side = float(sigma);

    DVec4 result = dvzero();

    [MaxIters(4)]

    for (int A = 0; A < DIM; ++A)
    {
        float a = float(A + 1);
        float phase = phase0 + orientation * 0.29 * a;
        float base = 0.52 * h[A] + orientation * 0.11 * a;
        float slope = orientation * (0.055 + 0.018 * a);
        float amplitude = 0.13 + 0.015 * a;
        float frequency = 0.47 + 0.19 * a;

        D1 value = D(base, 0.0);
        value = dadd(value, dscale(slope, z));
        value = dadd(
            value,
            dscale(
                amplitude,
                dsin(dadd(dscale(frequency, z), D(phase, 0.0)))
            )
        );

        // Explicit one-sided Carter splitting.
        value = dadd(
            value,
            dscale(
                PARAM_branchSplit * side * h[A],
                dtanh(dadd(dscale(1.65, z), D(phase, 0.0)))
            )
        );

        result.v[A] = value.v;
        result.dz[A] = value.dz;
    }

    return result;
}

[Differentiable]
DMat4 metricField(D1 z, int sigma)
{
    float phase = vertexPhase(H_G);
    float side = float(sigma);
    DMat4 G = dmzero();

    D1 g00 = dneg(dadd(
        D(1.28 + 0.015 * side, 0.0),
        dscale(0.075, dtanh(dadd(dscale(0.72, z), D(phase, 0.0))))
    ));

    D1 g11 = dadd(
        D(1.04, 0.0),
        dscale(0.045, dsin(dadd(dscale(0.61, z), D(phase * 0.7, 0.0))))
    );

    D1 g22 = dadd(
        D(1.09, 0.0),
        dscale(0.038, dcos(dadd(dscale(0.53, z), D(phase * 0.9, 0.0))))
    );

    D1 g33 = dadd(
        D(1.13, 0.0),
        dscale(0.042, dtanh(dadd(dscale(0.83, z), D(phase * 1.1, 0.0))))
    );

    dmset(G, 0, 0, g00);
    dmset(G, 1, 1, g11);
    dmset(G, 2, 2, g22);
    dmset(G, 3, 3, g33);

    D1 g01 = dscale(
        0.014,
        dsin(dadd(dscale(0.39, z), D(phase + 0.2 * side, 0.0)))
    );

    D1 g12 = dscale(
        0.011,
        dcos(dadd(dscale(0.44, z), D(phase * 0.8, 0.0)))
    );

    D1 g23 = dscale(
        0.009,
        dsin(dadd(dscale(0.57, z), D(phase * 1.2 - 0.15 * side, 0.0)))
    );

    dmset(G, 0, 1, g01);
    dmset(G, 1, 0, g01);
    dmset(G, 1, 2, g12);
    dmset(G, 2, 1, g12);
    dmset(G, 2, 3, g23);
    dmset(G, 3, 2, g23);

    return G;
}

[Differentiable]
D1 gammaField(D1 z, int sigma)
{
    float4 h = tesseractCoord(H_GAMMA);
    float phase = vertexPhase(H_GAMMA);
    float side = float(sigma);

    D1 gamma = D(1.0 + 0.035 * side * h.w, 0.0);
    gamma = dadd(
        gamma,
        dscale(0.19, dtanh(dadd(dscale(0.78, z), D(phase, 0.0))))
    );
    gamma = dadd(
        gamma,
        dscale(0.035 * h.y, dsin(dadd(dscale(0.31, z), D(phase * 0.5, 0.0))))
    );

    return gamma;
}

[Differentiable]
D1 invariantPlus(DMat4 G, DVec4 JPlus)
{
    return dneg(dvdot(JPlus, dmvec(G, JPlus)));
}

[Differentiable]
D1 invariantMinus(DMat4 G, DVec4 JMinus)
{
    return dneg(dvdot(JMinus, dmvec(G, JMinus)));
}

[Differentiable]
D1 invariantCross(DMat4 G, DVec4 JPlus, DVec4 JMinus)
{
    return dneg(dvdot(JPlus, dmvec(G, JMinus)));
}

// -----------------------------------------------------------------------------
// Bounded KKT sector.
// -----------------------------------------------------------------------------

struct DualBoxQP : IDifferentiable
{
    float H[KKT_N * KKT_N];
    float dH[KKT_N * KKT_N];

    float f[KKT_N];
    float df[KKT_N];

    float lo[KKT_N];
    float dlo[KKT_N];

    float hi[KKT_N];
    float dhi[KKT_N];
};

struct KKTResult : IDifferentiable
{
    float x[KKT_N];
    float dx[KKT_N];
    int activeSet[KKT_N];

    int activeMask;
    float objective;
    float feasibilityMargin;
    float conditionEstimate;
    int valid;
};

int midx(int row, int column)
{
    return row * KKT_N + column;
}

[Differentiable]
bool solveLinear(
    in float Ain[KKT_N * KKT_N],
    in float bin[KKT_N],
    out float x[KKT_N]
)
{
    [MaxIters(4)]
    for (int i = 0; i < KKT_N; ++i)
        x[i] = 0.0;

    float A[KKT_N * KKT_N] = {};
    float b[KKT_N] = {};

    [MaxIters(16)]

    for (int i = 0; i < KKT_N * KKT_N; ++i)
        A[i] = Ain[i];

    [MaxIters(4)]

    for (int i = 0; i < KKT_N; ++i)
        b[i] = bin[i];

    [MaxIters(4)]

    for (int column = 0; column < KKT_N; ++column)
    {
        int pivot = column;
        float bestPivot = abs(A[midx(column, column)]);

        [MaxIters(3)]

        for (int row = column + 1; row < KKT_N; ++row)
        {
            float candidate = abs(A[midx(row, column)]);
            if (candidate > bestPivot)
            {
                bestPivot = candidate;
                pivot = row;
            }
        }

        if (bestPivot < 1.0e-10)
            return false;

        if (pivot != column)
        {
            [MaxIters(4)]
            for (int j = 0; j < KKT_N; ++j)
            {
                float temporary = A[midx(column, j)];
                A[midx(column, j)] = A[midx(pivot, j)];
                A[midx(pivot, j)] = temporary;
            }

            float temporaryB = b[column];
            b[column] = b[pivot];
            b[pivot] = temporaryB;
        }

        float inversePivot = 1.0 / A[midx(column, column)];

        [MaxIters(4)]

        for (int j = column; j < KKT_N; ++j)
            A[midx(column, j)] *= inversePivot;

        b[column] *= inversePivot;

        [MaxIters(4)]

        for (int row = 0; row < KKT_N; ++row)
        {
            if (row == column)
                continue;

            float factor = A[midx(row, column)];

            [MaxIters(4)]

            for (int j = column; j < KKT_N; ++j)
                A[midx(row, j)] -= factor * A[midx(column, j)];

            b[row] -= factor * b[column];
        }
    }

    [MaxIters(4)]

    for (int i = 0; i < KKT_N; ++i)
        x[i] = b[i];

    return true;
}

[Differentiable]
float matrixConditionEstimate(in float A[KKT_N * KKT_N])
{
    float normA = 0.0;
    [MaxIters(4)]
    for (int row = 0; row < KKT_N; ++row)
    {
        float rowSum = 0.0;
        [MaxIters(4)]
        for (int column = 0; column < KKT_N; ++column)
            rowSum += abs(A[midx(row, column)]);
        normA = max(normA, rowSum);
    }

    float inverse[KKT_N * KKT_N] = {};
    [MaxIters(16)]
    for (int i = 0; i < KKT_N * KKT_N; ++i)
        inverse[i] = 0.0;

    [MaxIters(4)]

    for (int column = 0; column < KKT_N; ++column)
    {
        float rhs[KKT_N] = {};
        float solution[KKT_N] = {};
        [MaxIters(4)]
        for (int row = 0; row < KKT_N; ++row)
            rhs[row] = row == column ? 1.0 : 0.0;
        if (!solveLinear(A, rhs, solution))
            return 3.402823e38;
        [MaxIters(4)]
        for (int row = 0; row < KKT_N; ++row)
            inverse[midx(row, column)] = solution[row];
    }

    float normInverse = 0.0;
    [MaxIters(4)]
    for (int row = 0; row < KKT_N; ++row)
    {
        float rowSum = 0.0;
        [MaxIters(4)]
        for (int column = 0; column < KKT_N; ++column)
            rowSum += abs(inverse[midx(row, column)]);
        normInverse = max(normInverse, rowSum);
    }
    return normA * normInverse;
}

int integerPower3(int exponent)
{
    int result = 1;
    [MaxIters(4)]
    for (int i = 0; i < exponent; ++i)
        result *= 3;
    return result;
}

int activeState(int mask, int variable)
{
    return ((mask / integerPower3(variable)) % 3) - 1;
}

int activePreference(int state, int sigma)
{
    // Infinitesimal one-sided tie convention at exact KKT boundaries.
    if (sigma > 0)
    {
        if (state > 0) return 0;
        if (state == 0) return 1;
        return 2;
    }

    if (state < 0) return 0;
    if (state == 0) return 1;
    return 2;
}

int activeTieScore(int mask, int sigma)
{
    int score = 0;
    int place = 1;

    [MaxIters(4)]

    for (int i = 0; i < KKT_N; ++i)
    {
        score += activePreference(activeState(mask, i), sigma) * place;
        place *= 3;
    }

    return score;
}

[Differentiable]
DualBoxQP buildKKTProblem(
    D1 z,
    D1 Gamma,
    D1 XPlus,
    D1 XMinus,
    D1 XCross,
    int sigma
)
{
    DualBoxQP qp = {};
    D1 Hd[KKT_N * KKT_N] = {};
    D1 target[KKT_N] = {};
    D1 lo[KKT_N] = {};
    D1 hi[KKT_N] = {};

    float phase = vertexPhase(H_KKT);
    float side = float(sigma);

    target[0] = dadd(
        dscale(
            0.56,
            dtanh(dadd(
                dadd(dscale(0.43, XCross), dscale(0.24, Gamma)),
                dadd(dscale(0.12, z), D(phase * 0.08, 0.0))
            ))
        ),
        D(0.075 * side, 0.0)
    );

    target[1] = dscale(
        0.43,
        dtanh(dadd(
            dsub(dscale(0.36, XPlus), dscale(0.11, XMinus)),
            D(phase * 0.04, 0.0)
        ))
    );

    target[2] = dscale(
        0.43,
        dtanh(dadd(
            dsub(dscale(0.36, XMinus), dscale(0.11, XPlus)),
            D(-phase * 0.03, 0.0)
        ))
    );

    target[3] = dadd(
        dscale(
            0.31,
            dtanh(dadd(
                dadd(dscale(0.29, dsub(XPlus, XMinus)), dscale(0.18, z)),
                D(phase * 0.06, 0.0)
            ))
        ),
        D(0.045 * side, 0.0)
    );

    lo[0] = dadd(D(-0.32, 0.0), dscale(0.025, dtanh(dscale(0.48, z))));
    hi[0] = dadd(D( 0.32, 0.0), dscale(0.025, dtanh(dscale(0.48, z))));

    lo[1] = dadd(D(-0.46, 0.0), dscale(0.018, dsin(dadd(dscale(0.37, z), D(phase, 0.0)))));
    hi[1] = dadd(D( 0.46, 0.0), dscale(0.018, dsin(dadd(dscale(0.37, z), D(phase, 0.0)))));

    lo[2] = dadd(D(-0.46, 0.0), dscale(0.017, dcos(dadd(dscale(0.41, z), D(phase * 0.7, 0.0)))));
    hi[2] = dadd(D( 0.46, 0.0), dscale(0.017, dcos(dadd(dscale(0.41, z), D(phase * 0.7, 0.0)))));

    lo[3] = dadd(D(-0.40, 0.0), dscale(0.020, dtanh(dadd(dscale(0.52, z), D(phase * 0.5, 0.0)))));
    hi[3] = dadd(D( 0.40, 0.0), dscale(0.020, dtanh(dadd(dscale(0.52, z), D(phase * 0.5, 0.0)))));

    [MaxIters(4)]

    for (int i = 0; i < KKT_N; ++i)
    {
        [MaxIters(4)]
        for (int j = 0; j < KKT_N; ++j)
        {
            D1 hij;

            if (i == j)
            {
                float fi = float(i + 1);
                hij = D(1.42 + 0.24 * fi, 0.0);
                hij = dadd(hij, dscale(0.070, dsquare(Gamma)));
                hij = dadd(hij, dscale(0.022 * fi, dsmoothAbs(XCross)));
            }
            else
            {
                float separation = float(abs(i - j) + 1);
                float frequency = 0.28 + 0.055 * float(i + j + 1);
                hij = dscale(
                    0.055 / separation,
                    dtanh(dadd(dscale(frequency, z), D(phase * 0.13 * float(i + j + 1), 0.0)))
                );
            }

            Hd[midx(i, j)] = hij;
            qp.H[midx(i, j)] = hij.v;
            qp.dH[midx(i, j)] = hij.dz;
        }
    }

    // f = -H target, so the unconstrained minimizer is target.
    [MaxIters(4)]
    for (int i = 0; i < KKT_N; ++i)
    {
        D1 fi = D(0.0, 0.0);

        [MaxIters(4)]

        for (int j = 0; j < KKT_N; ++j)
            fi = dsub(fi, dmul(Hd[midx(i, j)], target[j]));

        qp.f[i] = fi.v;
        qp.df[i] = fi.dz;
        qp.lo[i] = lo[i].v;
        qp.dlo[i] = lo[i].dz;
        qp.hi[i] = hi[i].v;
        qp.dhi[i] = hi[i].dz;
    }

    return qp;
}

[Differentiable]
KKTResult solveBoundedKKT(DualBoxQP qp, int sigma)
{
    const float FEASIBILITY_TOL = 2.0e-5;
    const float OBJECTIVE_TOL = 1.0e-7;

    KKTResult best = {};
    best.valid = 0;
    best.objective = 3.402823e38;
    best.feasibilityMargin = -3.402823e38;
    best.conditionEstimate = 3.402823e38;
    best.activeMask = -1;
    int bestTieScore = 2147483647;

    [MaxIters(4)]

    for (int i = 0; i < KKT_N; ++i)
    {
        best.x[i] = 0.0;
        best.dx[i] = 0.0;
        best.activeSet[i] = 0;
    }

    [MaxIters(81)]

    for (int mask = 0; mask < KKT_STATES; ++mask)
    {
        float A[KKT_N * KKT_N] = {};
        float b[KKT_N] = {};
        int states[KKT_N] = {};

        [MaxIters(16)]

        for (int i = 0; i < KKT_N * KKT_N; ++i)
            A[i] = 0.0;

        [MaxIters(4)]

        for (int i = 0; i < KKT_N; ++i)
        {
            states[i] = activeState(mask, i);

            if (states[i] == 0)
            {
                [MaxIters(4)]
                for (int j = 0; j < KKT_N; ++j)
                    A[midx(i, j)] = qp.H[midx(i, j)];

                b[i] = -qp.f[i];
            }
            else
            {
                A[midx(i, i)] = 1.0;

                bool lower = states[i] < 0;
                b[i] = lower ? qp.lo[i] : qp.hi[i];
            }
        }

        float x[KKT_N] = {};
        if (!solveLinear(A, b, x))
            continue;

        bool valid = true;
        float gradient[KKT_N] = {};

        [MaxIters(4)]

        for (int i = 0; i < KKT_N; ++i)
        {
            gradient[i] = qp.f[i];

            [MaxIters(4)]

            for (int j = 0; j < KKT_N; ++j)
                gradient[i] += qp.H[midx(i, j)] * x[j];

            if (x[i] < qp.lo[i] - FEASIBILITY_TOL ||
                x[i] > qp.hi[i] + FEASIBILITY_TOL)
            {
                valid = false;
            }

            if (states[i] == 0 && abs(gradient[i]) > FEASIBILITY_TOL)
                valid = false;

            // At a lower bound, grad >= 0. At an upper bound, grad <= 0.
            if (states[i] < 0 && gradient[i] < -FEASIBILITY_TOL)
                valid = false;

            if (states[i] > 0 && gradient[i] > FEASIBILITY_TOL)
                valid = false;
        }

        if (!valid)
            continue;

        float objective = 0.0;

        [MaxIters(4)]

        for (int i = 0; i < KKT_N; ++i)
        {
            objective += qp.f[i] * x[i];

            [MaxIters(4)]

            for (int j = 0; j < KKT_N; ++j)
                objective += 0.5 * x[i] * qp.H[midx(i, j)] * x[j];
        }

        int tieScore = activeTieScore(mask, sigma);
        bool improves = objective < best.objective - OBJECTIVE_TOL;
        bool tiedAndPreferred =
            abs(objective - best.objective) <= OBJECTIVE_TOL &&
            tieScore < bestTieScore;

        if (!improves && !tiedAndPreferred)
            continue;

        best.valid = 1;
        best.objective = objective;
        best.activeMask = mask;
        bestTieScore = tieScore;

        [MaxIters(4)]

        for (int i = 0; i < KKT_N; ++i)
        {
            best.x[i] = x[i];
            best.activeSet[i] = states[i];
        }
    }

    if (best.valid != 0)
    {
        // Rebuild only the winning system.  This is the selected KKT residual
        // used by both the tangent and its transpose-VJP counterpart:
        // A dx = db - dA x and A^T lambda = dL/dx.
        float A[KKT_N * KKT_N] = {};
        float dA[KKT_N * KKT_N] = {};
        float db[KKT_N] = {};
        [MaxIters(16)]
        for (int i = 0; i < KKT_N * KKT_N; ++i)
        {
            A[i] = 0.0;
            dA[i] = 0.0;
        }
        [MaxIters(4)]
        for (int i = 0; i < KKT_N; ++i)
        {
            if (best.activeSet[i] == 0)
            {
                [MaxIters(4)]
                for (int j = 0; j < KKT_N; ++j)
                {
                    A[midx(i, j)] = qp.H[midx(i, j)];
                    dA[midx(i, j)] = qp.dH[midx(i, j)];
                }
                db[i] = -qp.df[i];
            }
            else
            {
                A[midx(i, i)] = 1.0;
                bool lower = best.activeSet[i] < 0;
                db[i] = lower ? qp.dlo[i] : qp.dhi[i];
            }
        }

        float derivativeRhs[KKT_N] = {};
        [MaxIters(4)]
        for (int i = 0; i < KKT_N; ++i)
        {
            derivativeRhs[i] = db[i];
            [MaxIters(4)]
            for (int j = 0; j < KKT_N; ++j)
                derivativeRhs[i] -= dA[midx(i, j)] * best.x[j];
        }
        float dx[KKT_N] = {};
        if (!solveLinear(A, derivativeRhs, dx))
        {
            best.valid = 0;
        }
        else
        {
            best.feasibilityMargin = 3.402823e38;
            best.conditionEstimate = matrixConditionEstimate(A);
            [MaxIters(4)]
            for (int i = 0; i < KKT_N; ++i)
            {
                best.dx[i] = dx[i];
                float gradient = qp.f[i];
                [MaxIters(4)]
                for (int j = 0; j < KKT_N; ++j)
                    gradient += qp.H[midx(i, j)] * best.x[j];

                float margin;
                if (best.activeSet[i] < 0)
                    margin = gradient;
                else if (best.activeSet[i] > 0)
                    margin = -gradient;
                else
                    margin = min(
                        best.x[i] - qp.lo[i],
                        qp.hi[i] - best.x[i]
                    );
                best.feasibilityMargin =
                    min(best.feasibilityMargin, margin);
            }
        }
    }

    return best;
}

[Differentiable]
D1 kktVariable(KKTResult result, int variable)
{
    return D(result.x[variable], result.dx[variable]);
}

[Differentiable]
DVec4 kktTheta(KKTResult result)
{
    DVec4 theta = dvzero();

    theta.v.x = result.x[1];
    theta.v.y = result.x[2];
    theta.v.z = result.x[3];

    theta.dz.x = result.dx[1];
    theta.dz.y = result.dx[2];
    theta.dz.z = result.dx[3];

    return theta;
}

// -----------------------------------------------------------------------------
// Constitutive Lambda and Carter closure.
// -----------------------------------------------------------------------------

struct LambdaResult : IDifferentiable
{
    D1 value;
    D1 dXPlus;
    D1 dXMinus;
    D1 dXCross;
};

[Differentiable]
LambdaResult analyticLambdaField(
    D1 XPlus,
    D1 XMinus,
    D1 XCross,
    D1 Gamma,
    D1 qLimited,
    IDiffTensor<float, 1> weights
)
{
    float phase = vertexPhase(H_LAMBDA);

    D1 coefficientP = dadd(
        D(PARAM_lambdaP, 0.0),
        dadd(dscale(0.050, qLimited), dscale(0.020, Gamma))
    );

    D1 coefficientM = dadd(
        D(PARAM_lambdaM, 0.0),
        dadd(dscale(-0.040, qLimited), dscale(0.017, Gamma))
    );

    D1 coefficientC = dadd(
        D(PARAM_lambdaC, 0.0),
        dadd(dscale(0.030, Gamma), D(0.004 * phase, 0.0))
    );

    D1 value = D(PARAM_lambda0, 0.0);
    value = dadd(value, dmul(coefficientP, XPlus));
    value = dadd(value, dmul(coefficientM, XMinus));
    value = dadd(value, dmul(coefficientC, XCross));

    value = dadd(value, dscale(0.5 * PARAM_lambdaPP, dsquare(XPlus)));
    value = dadd(value, dscale(0.5 * PARAM_lambdaMM, dsquare(XMinus)));
    value = dadd(value, dscale(0.5 * PARAM_lambdaCC, dsquare(XCross)));
    value = dadd(value, dscale(PARAM_lambdaPM, dmul(XPlus, XMinus)));

    value = dadd(value, dscale(PARAM_lambdaGamma, Gamma));
    value = dadd(value, dscale(PARAM_lambdaQ, qLimited));
    value = dadd(value, dscale(PARAM_lambdaGammaQ, dmul(Gamma, qLimited)));

    LambdaResult result;
    result.value = value;

    result.dXPlus = dadd(
        coefficientP,
        dadd(
            dscale(PARAM_lambdaPP, XPlus),
            dscale(PARAM_lambdaPM, XMinus)
        )
    );

    result.dXMinus = dadd(
        coefficientM,
        dadd(
            dscale(PARAM_lambdaMM, XMinus),
            dscale(PARAM_lambdaPM, XPlus)
        )
    );

    result.dXCross = dadd(
        coefficientC,
        dscale(PARAM_lambdaCC, XCross)
    );

    return result;
}

[Differentiable]
LambdaResult neuralLambdaField(
    D1 XPlus,
    D1 XMinus,
    D1 XCross,
    D1 Gamma,
    D1 qLimited,
    IDiffTensor<float, 1> weights
)
{
    // The learned scalar is a linear path plus a two-layer tanh residual MLP.
    // D1 forward propagation carries d/dz. Reverse propagation in D1 arithmetic
    // returns both each Carter partial and its directional d/dz derivative.
    D1 inputValue[5];
    D1 hidden1[MASTER_WIDTH] = {};
    D1 hidden2[MASTER_WIDTH] = {};
    D1 adjointHidden1[MASTER_WIDTH] = {};
    D1 adjointPre1[MASTER_WIDTH] = {};
    D1 adjointHidden2[MASTER_WIDTH] = {};
    D1 adjointPre2[MASTER_WIDTH] = {};
    D1 adjointInput[5];

    inputValue[0] = XPlus;
    inputValue[1] = XMinus;
    inputValue[2] = XCross;
    inputValue[3] = Gamma;
    inputValue[4] = qLimited;

    D1 outputValue = D(weights[MW_SKIP_B], 0.0);
    [MaxIters(5)]
    for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
    {
        outputValue = dadd(
            outputValue,
            dscale(weights[MW_SKIP_W + inputIndex], inputValue[inputIndex])
        );
    }

    [MaxIters(32)]

    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        D1 preactivation = D(weights[MW_L1_B + neuron], 0.0);
        [MaxIters(5)]
        for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
        {
            int weightIndex = MW_L1_W + neuron * 5 + inputIndex;
            preactivation = dadd(
                preactivation,
                dscale(weights[weightIndex], inputValue[inputIndex])
            );
        }
        hidden1[neuron] = dtanh(preactivation);
    }

    [MaxIters(32)]

    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        D1 preactivation = D(weights[MW_L2_B + neuron], 0.0);
        [MaxIters(32)]
        for (int inputIndex = 0; inputIndex < MASTER_WIDTH; ++inputIndex)
        {
            int weightIndex =
                MW_L2_W + neuron * MASTER_WIDTH + inputIndex;
            preactivation = dadd(
                preactivation,
                dscale(weights[weightIndex], hidden1[inputIndex])
            );
        }
        hidden2[neuron] = dtanh(preactivation);
    }

    outputValue = dadd(outputValue, D(weights[MW_L3_B], 0.0));
    [MaxIters(32)]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        outputValue = dadd(
            outputValue,
            dscale(weights[MW_L3_W + neuron], hidden2[neuron])
        );
    }

    // Reverse mode over the MLP, evaluated with dual adjoints.
    [MaxIters(32)]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        adjointHidden2[neuron] = D(weights[MW_L3_W + neuron], 0.0);
        adjointPre2[neuron] = dmul(
            adjointHidden2[neuron],
            dsub(D(1.0, 0.0), dsquare(hidden2[neuron]))
        );
    }

    [MaxIters(32)]

    for (int inputIndex = 0; inputIndex < MASTER_WIDTH; ++inputIndex)
    {
        D1 adjoint = D(0.0, 0.0);
        [MaxIters(32)]
        for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
        {
            int weightIndex =
                MW_L2_W + neuron * MASTER_WIDTH + inputIndex;
            adjoint = dadd(
                adjoint,
                dscale(weights[weightIndex], adjointPre2[neuron])
            );
        }
        adjointHidden1[inputIndex] = adjoint;
        adjointPre1[inputIndex] = dmul(
            adjointHidden1[inputIndex],
            dsub(D(1.0, 0.0), dsquare(hidden1[inputIndex]))
        );
    }

    [MaxIters(5)]

    for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
    {
        D1 adjoint = D(weights[MW_SKIP_W + inputIndex], 0.0);
        [MaxIters(32)]
        for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
        {
            int weightIndex = MW_L1_W + neuron * 5 + inputIndex;
            adjoint = dadd(
                adjoint,
                dscale(weights[weightIndex], adjointPre1[neuron])
            );
        }
        adjointInput[inputIndex] = adjoint;
    }

    LambdaResult result;
    result.value = outputValue;
    result.dXPlus = adjointInput[0];
    result.dXMinus = adjointInput[1];
    result.dXCross = adjointInput[2];
    return result;
}

[Differentiable]
LambdaResult lambdaField(
    D1 XPlus,
    D1 XMinus,
    D1 XCross,
    D1 Gamma,
    D1 qLimited,
    IDiffTensor<float, 1> weights
)
{
    if (1u != 0u)
        return neuralLambdaField(XPlus, XMinus, XCross, Gamma, qLimited, weights);

    return analyticLambdaField(XPlus, XMinus, XCross, Gamma, qLimited, weights);
}

[Differentiable]
DMat4 stressTensor(
    D1 Psi,
    DVec4 JPlus,
    DVec4 JMinus,
    DVec4 piPlus,
    DVec4 piMinus
)
{
    DMat4 T = dmzero();

    [MaxIters(4)]

    for (int A = 0; A < DIM; ++A)
    {
        [MaxIters(4)]
        for (int B = 0; B < DIM; ++B)
        {
            float delta = A == B ? 1.0 : 0.0;

            D1 component = D(Psi.v * delta, Psi.dz * delta);
            component = dadd(
                component,
                dmul(
                    D(JPlus.v[A], JPlus.dz[A]),
                    D(piPlus.v[B], piPlus.dz[B])
                )
            );
            component = dadd(
                component,
                dmul(
                    D(JMinus.v[A], JMinus.dz[A]),
                    D(piMinus.v[B], piMinus.dz[B])
                )
            );

            dmset(T, A, B, component);
        }
    }

    return T;
}

int widx(int A, int B, int C, int Dindex)
{
    return (((A * DIM + B) * DIM + C) * DIM + Dindex);
}

[Differentiable]
float stressWeight(int A, int B, int C, int Dindex)
{
    if (0u != 0u)
        return 0.0;

    return (A == C && B == Dindex) ? 1.0 : 0.0;
}

[Differentiable]
D1 weightedStressNorm(DMat4 T)
{
    D1 result = D(0.0, 0.0);

    [MaxIters(4)]

    for (int A = 0; A < DIM; ++A)
    [MaxIters(4)]
    for (int B = 0; B < DIM; ++B)
    [MaxIters(4)]
    for (int C = 0; C < DIM; ++C)
    [MaxIters(4)]
    for (int Dindex = 0; Dindex < DIM; ++Dindex)
    {
        D1 TAB = D(
            T.v[B][A] - TARGET_T[B][A],
            T.dz[B][A]
        );

        D1 TCD = D(
            T.v[Dindex][C] - TARGET_T[Dindex][C],
            T.dz[Dindex][C]
        );

        result = dadd(
            result,
            dscale(
                stressWeight(A, B, C, Dindex),
                dmul(TAB, TCD)
            )
        );
    }

    return result;
}

[Differentiable]
D1 energyField(
    D1 XPlus,
    D1 XMinus,
    D1 XCross,
    D1 Gamma,
    DVec4 theta
)
{
    float phase = vertexPhase(H_ENERGY);

    D1 energy = dscale(
        0.5,
        dadd(dsmoothAbs(XPlus), dsmoothAbs(XMinus))
    );

    energy = dadd(energy, dscale(0.25, dsquare(XCross)));
    energy = dadd(energy, dscale(0.10, dsquare(dsub(Gamma, D(1.0, 0.0)))));

    [MaxIters(3)]

    for (int i = 0; i < 3; ++i)
        energy = dadd(energy, dscale(0.05, dsquare(dvcomponent(theta, i))));

    energy = dadd(energy, D(0.0025 * phase, 0.0));
    return energy;
}

[Differentiable]
D1 limitedJField(D1 z, D1 qLimited, DVec4 theta)
{
    float phase = vertexPhase(H_J_LIMITED);
    D1 theta0 = dvcomponent(theta, 0);
    D1 theta1 = dvcomponent(theta, 1);
    D1 theta2 = dvcomponent(theta, 2);

    D1 result = qLimited;
    result = dadd(result, dscale(0.50, dmul(theta0, theta1)));
    result = dsub(result, dscale(0.25, theta2));
    result = dadd(
        result,
        dscale(
            0.10,
            dsin(dadd(dadd(z, theta2), D(phase * 0.1, 0.0)))
        )
    );

    return result;
}

[Differentiable]
D1 residualField(
    D1 Gamma,
    D1 qLimited,
    DVec4 theta,
    D1 XCross
)
{
    float phase = vertexPhase(H_RESIDUAL);

    D1 meanTheta = dscale(
        1.0 / 3.0,
        dadd(
            dadd(dvcomponent(theta, 0), dvcomponent(theta, 1)),
            dvcomponent(theta, 2)
        )
    );

    D1 result = dadd(Gamma, dmul(qLimited, XCross));
    result = dsub(result, meanTheta);
    result = dadd(result, D(0.003 * phase, 0.0));
    return result;
}

[Differentiable]
D1 freeEnergyField(
    D1 energy,
    DMat4 T,
    DVec4 piPlus,
    DVec4 piMinus,
    DMat4 G,
    D1 entrainmentA,
    D1 XCross
)
{
    D1 stressTerm = weightedStressNorm(T);
    DVec4 deltaPi = dvsub(piPlus, piMinus);
    D1 momentumTerm = dvdot(deltaPi, dmvec(G, deltaPi));
    D1 entrainmentTerm = dmul(entrainmentA, XCross);

    D1 result = dscale(PARAM_omegaE, energy);
    result = dadd(result, dscale(0.5 * PARAM_omegaT, stressTerm));
    result = dadd(result, dscale(0.5 * PARAM_omegaPi, momentumTerm));
    result = dadd(result, dscale(PARAM_omegaA, entrainmentTerm));
    return result;
}

// -----------------------------------------------------------------------------
// Complete tesseract state.
// -----------------------------------------------------------------------------

struct TesseractState : IDifferentiable
{
    D1 scalar[16];
    DVec4 vector[16];
    DMat4 matrix4[16];

    KKTResult kkt;
    DVec4 piPlus;
    DVec4 piMinus;
    D1 BPlus;
    D1 BMinus;
    D1 entrainmentA;
    D1 Psi;

    uint classifierMask;
    float classifierMargin[4];
};

[Differentiable]
TesseractState evaluateTesseract(D1 z, int sigma, IDiffTensor<float, 1> weights)
{
    TesseractState state;

    [MaxIters(16)]

    for (int i = 0; i < 16; ++i)
    {
        state.scalar[i] = D(0.0, 0.0);
        state.vector[i] = dvzero();
        state.matrix4[i] = dmzero();
    }

    // Primitive Carter face.
    state.vector[H_J_PLUS] = currentField(z, H_J_PLUS, sigma, +1.0);
    state.vector[H_J_MINUS] = currentField(z, H_J_MINUS, sigma, -1.0);
    state.matrix4[H_G] = metricField(z, sigma);
    state.scalar[H_GAMMA] = gammaField(z, sigma);

    // Invariant face.
    state.scalar[H_X_PLUS] = invariantPlus(
        state.matrix4[H_G],
        state.vector[H_J_PLUS]
    );

    state.scalar[H_X_MINUS] = invariantMinus(
        state.matrix4[H_G],
        state.vector[H_J_MINUS]
    );

    state.scalar[H_X_CROSS] = invariantCross(
        state.matrix4[H_G],
        state.vector[H_J_PLUS],
        state.vector[H_J_MINUS]
    );

    // Bounded KKT face and its exact active-branch pullback.
    DualBoxQP qp = buildKKTProblem(
        z,
        state.scalar[H_GAMMA],
        state.scalar[H_X_PLUS],
        state.scalar[H_X_MINUS],
        state.scalar[H_X_CROSS],
        sigma
    );

    state.kkt = no_diff(solveBoundedKKT(qp, sigma));
    state.scalar[H_KKT] = D(state.kkt.objective, 0.0);
    state.scalar[H_Q_LIMITED] = kktVariable(state.kkt, 0);
    state.vector[H_THETA_STAR] = kktTheta(state.kkt);

    // Constitutive vertex.
    LambdaResult lambda = lambdaField(
        state.scalar[H_X_PLUS],
        state.scalar[H_X_MINUS],
        state.scalar[H_X_CROSS],
        state.scalar[H_GAMMA],
        state.scalar[H_Q_LIMITED],
        weights
    );

    state.scalar[H_LAMBDA] = lambda.value;

    state.BPlus = dscale(-2.0, lambda.dXPlus);
    state.BMinus = dscale(-2.0, lambda.dXMinus);
    state.entrainmentA = dscale(-1.0, lambda.dXCross);

    DVec4 JPlusCov = dmvec(state.matrix4[H_G], state.vector[H_J_PLUS]);
    DVec4 JMinusCov = dmvec(state.matrix4[H_G], state.vector[H_J_MINUS]);

    state.piPlus = dvadd(
        dvscale(state.BPlus, JPlusCov),
        dvscale(state.entrainmentA, JMinusCov)
    );

    state.piMinus = dvadd(
        dvscale(state.BMinus, JMinusCov),
        dvscale(state.entrainmentA, JPlusCov)
    );

    state.Psi = lambda.value;
    state.Psi = dsub(
        state.Psi,
        dvdot(state.vector[H_J_PLUS], state.piPlus)
    );
    state.Psi = dsub(
        state.Psi,
        dvdot(state.vector[H_J_MINUS], state.piMinus)
    );

    // Carter closure vertex stores Psi, piPlus, and T. piMinus remains a typed
    // side-fiber because one vertex cannot hold two independent float4 payloads.
    state.scalar[H_CLOSURE] = state.Psi;
    state.vector[H_CLOSURE] = state.piPlus;
    state.matrix4[H_CLOSURE] = stressTensor(
        state.Psi,
        state.vector[H_J_PLUS],
        state.vector[H_J_MINUS],
        state.piPlus,
        state.piMinus
    );

    // Explicit missing definitions at their coordinates.
    state.scalar[H_ENERGY] = energyField(
        state.scalar[H_X_PLUS],
        state.scalar[H_X_MINUS],
        state.scalar[H_X_CROSS],
        state.scalar[H_GAMMA],
        state.vector[H_THETA_STAR]
    );

    state.scalar[H_J_LIMITED] = limitedJField(
        z,
        state.scalar[H_Q_LIMITED],
        state.vector[H_THETA_STAR]
    );

    state.scalar[H_RESIDUAL] = residualField(
        state.scalar[H_GAMMA],
        state.scalar[H_Q_LIMITED],
        state.vector[H_THETA_STAR],
        state.scalar[H_X_CROSS]
    );

    // Final free-energy vertex. Its .dz member is D^(C)_(z,sigma) M.
    state.scalar[H_M] = freeEnergyField(
        state.scalar[H_ENERGY],
        state.matrix4[H_CLOSURE],
        state.piPlus,
        state.piMinus,
        state.matrix4[H_G],
        state.entrainmentA,
        state.scalar[H_X_CROSS]
    );

    // Classifier certificate: four continuous switching margins.
    state.classifierMargin[0] = z.v;
    state.classifierMargin[1] = state.scalar[H_GAMMA].v - 1.0;
    state.classifierMargin[2] = state.scalar[H_X_CROSS].v;
    state.classifierMargin[3] =
        state.scalar[H_Q_LIMITED].v - state.vector[H_THETA_STAR].v.x;

    state.classifierMask = 0u;

    [MaxIters(4)]

    for (int i = 0; i < 4; ++i)
    {
        if (state.classifierMargin[i] >= 0.0)
            state.classifierMask |= (1u << uint(i));
    }

    return state;
}

// -----------------------------------------------------------------------------
// Boundary certificates and iota_n.
// -----------------------------------------------------------------------------

[Differentiable]
bool marginCrossed(float a, float b)
{
    float tolerance = max(PARAM_crossingTolerance, 0.0);

    return
        abs(a) <= tolerance ||
        abs(b) <= tolerance ||
        ((a < 0.0) != (b < 0.0));
}

[Differentiable]
bool certificateChanged(TesseractState a, TesseractState b)
{
    if (a.kkt.activeMask != b.kkt.activeMask)
        return true;

    if (a.classifierMask != b.classifierMask)
        return true;

    [MaxIters(4)]

    for (int i = 0; i < 4; ++i)
    {
        if (marginCrossed(a.classifierMargin[i], b.classifierMargin[i]))
            return true;
    }

    return false;
}

[Differentiable]
bool parityBoundaryCrossed(float z0, float z1)
{
    float u0 = 1.5 * (z0 + PARAM_c);
    float u1 = 1.5 * (z1 + PARAM_c);

    float a0 = abs(u0);
    float a1 = abs(u1);
    float maximum = max(a0, a1);
    float minimum = (u0 * u1 <= 0.0) ? 0.0 : min(a0, a1);
    float tolerance = max(PARAM_crossingTolerance, 0.0);

    int lowCell = int(floor(max(minimum - tolerance, 0.0)));
    int highCell = int(floor(maximum + tolerance));

    return highCell != lowCell;
}

[Differentiable]
bool boundaryCrossed(
    float z0,
    float z1,
    int sigma,
    TesseractState state0,
    IDiffTensor<float, 1> weights
)
{
    if (isnan(z1) || isinf(z1))
        return true;

    float midpoint = 0.5 * (z0 + z1);

    if (parityBoundaryCrossed(z0, z1))
        return true;

    TesseractState stateMid = evaluateTesseract(D(midpoint, 1.0), sigma, weights);
    TesseractState state1 = evaluateTesseract(D(z1, 1.0), sigma, weights);

    return
        certificateChanged(state0, stateMid) ||
        certificateChanged(stateMid, state1);
}

// -----------------------------------------------------------------------------
// Literal recurrence.
// -----------------------------------------------------------------------------

int selectedSigma(float z)
{
    if (0 > 0)
        return +1;

    if (0 < 0)
        return -1;

    return z >= 0.0 ? +1 : -1;
}

[Differentiable]
float recurrenceValue(
    float z,
    float dJLimited,
    float residual,
    float dM,
    float jumpM,
    float iota
)
{
    float u = 1.5 * (z + PARAM_c);
    float parity = no_diff(fmod(floor(abs(u)), 2.0));

    float normalization = sqrt(max(
        z * z + PARAM_c * PARAM_c + PARAM_epsilonN,
        SAFE_EPS
    ));

    float correction =
          PARAM_kappa * u
        - PARAM_alpha * dJLimited
        + PARAM_chi * residual
        - PARAM_zeta * (
              dM
            + PARAM_omegaDelta * iota * jumpM
          );

    return
          u * u
        + PARAM_c
        + (parity / normalization) * correction;
}



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
