import "slangpy";

static const float T33_GAMMA_AD = 1.6666666666666667;
static const float T33_CARRIER_K = 0.2;
static const float T33_CARRIER_SOUND_SPEED = 0.9;
static const float T33_CHEMICAL_SUSCEPTIBILITY = 10.0;
static const float T33_TARGET_FRACTION = 0.05;
static const float T33_ENTRAINMENT_LINEAR = 0.0;
static const float T33_ENTRAINMENT_QUADRATIC = 0.0;
static const float T33_ENTRAINMENT_SCALE_FOURTH = 1.0;
static const float T33_EPSILON = 1.0e-12;

[Differentiable]
float theory33LambdaValue(
    float nSquared,
    float dSquared,
    float xSquared,
    float entropy
)
{
    float n = sqrt(max(nSquared, T33_EPSILON));
    float d = sqrt(max(dSquared, T33_EPSILON));
    float pressure =
        exp((T33_GAMMA_AD - 1.0) * entropy)
        * pow(n, T33_GAMMA_AD);
    float thermal = n + pressure / (T33_GAMMA_AD - 1.0);
    float anchor = d - T33_TARGET_FRACTION * n;
    float rho0 =
        thermal
        + T33_CARRIER_K
        * pow(d, 1.0 + T33_CARRIER_SOUND_SPEED * T33_CARRIER_SOUND_SPEED)
        + 0.5 * anchor * anchor / T33_CHEMICAL_SUSCEPTIBILITY;
    float relative = xSquared - n * d;
    return
        -rho0
        - T33_ENTRAINMENT_LINEAR * relative
        - 0.5
        * T33_ENTRAINMENT_QUADRATIC
        * relative
        * relative
        / T33_ENTRAINMENT_SCALE_FOURTH;
}

// Native reverse-mode entrypoint for the analytic master. The four feature
// adjoints are the derivatives from which the Carter coefficients follow.
[Differentiable]
void theory33Master(
    int sample,
    IDiffTensor<float, 2> features,
    IWDiffTensor<float, 1> output
)
{
    output[sample] = theory33LambdaValue(
        features[sample, 0],
        features[sample, 1],
        features[sample, 2],
        features[sample, 3]
    );
}

float minimumEigenvalue2(float diagonal0, float offDiagonal, float diagonal1)
{
    float difference = diagonal0 - diagonal1;
    return 0.5 * (
        diagonal0
        + diagonal1
        - sqrt(max(
            difference * difference + 4.0 * offDiagonal * offDiagonal,
            0.0
        ))
    );
}

// Output columns:
// 0 Lambda, 1 B_N, 2 B_D, 3 A, 4 Psi, 5 gamma_relative,
// 6 minimum Legendre eigenvalue, 7 minimum thermodynamic eigenvalue.
void theory33Constitutive(
    int sample,
    ITensor<float, 2> features,
    IWTensor<float, 2> output
)
{
    float nSquared = features[sample, 0];
    float dSquared = features[sample, 1];
    float xSquared = features[sample, 2];
    float entropy = features[sample, 3];
    float n = sqrt(max(nSquared, T33_EPSILON));
    float d = sqrt(max(dSquared, T33_EPSILON));
    float soundSquared =
        T33_CARRIER_SOUND_SPEED * T33_CARRIER_SOUND_SPEED;
    float pressure =
        exp((T33_GAMMA_AD - 1.0) * entropy)
        * pow(n, T33_GAMMA_AD);
    float thermal = n + pressure / (T33_GAMMA_AD - 1.0);
    float anchor = d - T33_TARGET_FRACTION * n;
    float carrierEnergy =
        T33_CARRIER_K * pow(d, 1.0 + soundSquared);
    float chemicalEnergy =
        0.5 * anchor * anchor / T33_CHEMICAL_SUSCEPTIBILITY;
    float relative = xSquared - n * d;
    float entrainment =
        T33_ENTRAINMENT_LINEAR
        + T33_ENTRAINMENT_QUADRATIC
        * relative
        / T33_ENTRAINMENT_SCALE_FOURTH;
    float lambda =
        -(thermal + carrierEnergy + chemicalEnergy)
        - T33_ENTRAINMENT_LINEAR * relative
        - 0.5
        * T33_ENTRAINMENT_QUADRATIC
        * relative
        * relative
        / T33_ENTRAINMENT_SCALE_FOURTH;
    float chemicalBaryon =
        1.0
        + T33_GAMMA_AD
        * pressure
        / max((T33_GAMMA_AD - 1.0) * n, T33_EPSILON)
        - T33_TARGET_FRACTION
        * anchor
        / T33_CHEMICAL_SUSCEPTIBILITY;
    float chemicalCarrier =
        T33_CARRIER_K
        * (1.0 + soundSquared)
        * pow(d, soundSquared)
        + anchor / T33_CHEMICAL_SUSCEPTIBILITY;
    float bBaryon = (chemicalBaryon - entrainment * d) / n;
    float bCarrier = (chemicalCarrier - entrainment * n) / d;
    float generalizedPressure =
        lambda
        + bBaryon * nSquared
        + bCarrier * dSquared
        + 2.0 * entrainment * xSquared;
    float hessianNN =
        T33_GAMMA_AD * pressure / max(nSquared, T33_EPSILON)
        + T33_TARGET_FRACTION
        * T33_TARGET_FRACTION
        / T33_CHEMICAL_SUSCEPTIBILITY;
    float hessianND =
        -T33_TARGET_FRACTION / T33_CHEMICAL_SUSCEPTIBILITY;
    float hessianDD =
        T33_CARRIER_K
        * (1.0 + soundSquared)
        * soundSquared
        * pow(d, soundSquared - 1.0)
        + 1.0 / T33_CHEMICAL_SUSCEPTIBILITY;

    output[sample, 0] = lambda;
    output[sample, 1] = bBaryon;
    output[sample, 2] = bCarrier;
    output[sample, 3] = entrainment;
    output[sample, 4] = generalizedPressure;
    output[sample, 5] = xSquared / max(n * d, T33_EPSILON);
    output[sample, 6] = minimumEigenvalue2(
        bBaryon, entrainment, bCarrier
    );
    output[sample, 7] = minimumEigenvalue2(
        hessianNN, hessianND, hessianDD
    );
}
// Generated from theory33_frozen_variants.json. Do not hand edit.
static const float T33Q_ENERGY_SCALE = 1.5098707551524818e-08;
static const float T33Q_DRIFT_SCALE = 0.016;
static const float T33Q_PHASE_BASE = 0.0058757847839562387;
static const float T33Q_PHASE_T = 0.00022806400660869901;
static const float T33Q_PHASE_N = -1.1901127644338779e-05;
static const float T33Q_PHASE_F = -5.0946531669675872e-05;
static const float T33Q_DENSITY_MIN = 0.059999999999999998;
static const float T33Q_DENSITY_MAX = 0.17999999999999999;
static const float T33Q_TEMPERATURE_MIN =
    0.080000000000000002;
static const float T33Q_TEMPERATURE_MAX =
    0.80000000000000004;
static const float T33Q_FRACTION_MIN = 0.025000000000000001;
static const float T33Q_FRACTION_MAX = 0.089999999999999997;

static const int T33Q_NEGATIVE_LATTICE_COUNT = 24;
static const float T33Q_NEGATIVE_LATTICE[24] = { 0.82793560036921543, 0.77713494433857355, 0.72643303132087789, 0.67583082555323692, 0.62531025822103969, 0.57485067563342318, 0.52443229518432344, 0.47403588478003622, 0.42364183323789911, 0.37322948260733235, 0.32277676839316199, 0.27226004041122687, 0.22165395488243125, 0.17093139397713863, 0.12006343359106279, 0.069019423972471752, 0.017767207221196325, -0.033726626308172518, -0.085496990481370438, -0.13758220028017445, -0.19002905476767529, -0.24289959947106654, -0.2962614859785751, -0.35008058095332378 };
static const int T33Q_POSITIVE_LATTICE_COUNT = 24;
static const float T33Q_POSITIVE_LATTICE[24] = { 0.52132317442693021, 0.46891440941805873, 0.41633962153152138, 0.36348525164640677, 0.3102663771396269, 0.25661760946070367, 0.2024871880534842, 0.14783247797268065, 0.092617222756502637, 0.036809229658509288, -0.019620983034056252, -0.076701461439009563, -0.13445957764508845, -0.19292269330761933, -0.25211849833913974, -0.31207488432687613, -0.37281935042497288, -0.43437705756282741, -0.49676761868425656, -0.55999858823582405, -0.62405531120776381, -0.68888296377480862, -0.75435827566165725, -0.82024180879899844 };

[Differentiable]
float qualifiedSoftplus(float value)
{
    return max(value, 0.0) + log(1.0 + exp(-abs(value)));
}

[Differentiable]
float qualifiedSigmoid(float value)
{
    return 1.0 / (1.0 + exp(-value));
}

[Differentiable]
float qualifiedDriftNetwork(float y)
{
    return
        0.5 * 0.8920045273139886 * y * y
        + 0.0011765973754139952 * (qualifiedSoftplus(5 * (y - 0)) - 0.69314718055994529 - 2.5 * y)
        + 0.0012599340238446445 * (qualifiedSoftplus(5 * (y - 0.1111111111111111)) - 0.453463479995937 - 1.8228822037081964 * y)
        + 0.0014151318656033629 * (qualifiedSoftplus(5 * (y - 0.22222222222222221)) - 0.2845719819862717 - 1.2383190056953581 * y)
        + 0.001726781688101598 * (qualifiedSoftplus(5 * (y - 0.33333333333333331)) - 0.1730079888858391 - 0.79434552440457584 * y)
        + 0.0024327028435963361 * (qualifiedSoftplus(5 * (y - 0.44444444444444442)) - 0.10288868409747487 - 0.48886299925399657 * y)
        + 0.0042797212566394139 * (qualifiedSoftplus(5 * (y - 0.55555555555555558)) - 0.060320127483555752 - 0.29268451437183951 * y)
        + 0.0092652464100795549 * (qualifiedSoftplus(5 * (y - 0.66666666666666674)) - 0.035052416079463071 - 0.17222597833105577 * y)
        + 0.019721130669000966 * (qualifiedSoftplus(5 * (y - 0.77777777777777779)) - 0.020261419793971745 - 0.10028768268925206 * y)
        + 0.034892382216936912 * (qualifiedSoftplus(5 * (y - 0.88888888888888884)) - 0.011675207207204575 - 0.058036582226523158 * y)
        + 0.051762052865918319 * (qualifiedSoftplus(5 * (y - 1)) - 0.0067153484891180684 - 0.033464254621424279 * y);
}

[Differentiable]
float qualifiedPhaseThreshold(
    float density,
    float carrier,
    float entropy
)
{
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
}

[Differentiable]
float qualifiedTheory33LambdaValue(
    float nSquared,
    float dSquared,
    float xSquared,
    float entropy
)
{
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
}

[Differentiable]
void qualifiedTheory33Master(
    int sample,
    IDiffTensor<float, 2> features,
    IWDiffTensor<float, 1> output
)
{
    output[sample] = qualifiedTheory33LambdaValue(
        features[sample, 0],
        features[sample, 1],
        features[sample, 2],
        features[sample, 3]
    );
}

float qualifiedLattice(
    float coordinate,
    const float values[T33Q_NEGATIVE_LATTICE_COUNT]
)
{
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
}

float qualifiedKappaResidual(float z)
{
    float negativeCoordinate = clamp(
        2.0 * (z - (-1))
        / (-0.71666666666666667 - (-1))
        - 1.0,
        -1.0,
        1.0
    );
    float positiveCoordinate = clamp(
        2.0 * (z - 0.61666666666666659)
        / (1.05 - 0.61666666666666659)
        - 1.0,
        -1.0,
        1.0
    );
    return z < 0.0
        ? qualifiedLattice(negativeCoordinate, T33Q_NEGATIVE_LATTICE)
        : qualifiedLattice(positiveCoordinate, T33Q_POSITIVE_LATTICE);
}

void qualifiedKappa(
    int sample,
    ITensor<float, 1> z,
    IWTensor<float, 1> output
)
{
    output[sample] = qualifiedKappaResidual(z[sample]);
}

void qualifiedMobilityValues(
    float context0,
    float context1,
    float context2,
    out float l00,
    out float l01,
    out float l11
)
{
    float h0 = tanh(0.059266715655541063 + -0.24365937130771351 * context0 + 0.47526893344313664 * context1 + -0.11790292456428879 * context2);
    float h1 = tanh(-0.26428381511365562 + 0.11853925512835217 * context0 + 0.10758761394393036 * context1 + -0.023812046585708232 * context2);
    float h2 = tanh(-0.01767854372243893 + 0.023942152358771174 * context0 + -0.4721134243523169 * context1 + -0.003173505675986843 * context2);
    float h3 = tanh(0.034067579195687023 + 0.33622559433807159 * context0 + 0.32863899589880669 * context1 + 0.21364978947671939 * context2);
    float h4 = tanh(0.079809800430380023 + -0.099586330383672789 * context0 + 0.15037033490537163 * context1 + 0.19125664089180355 * context2);
    float h5 = tanh(-0.22945055349879501 + -0.41281955315944824 * context0 + 0.71495722465781775 * context1 + 0.064088018287894941 * context2);
    float h6 = tanh(0.055942646646867142 + 0.33214514375556325 * context0 + 0.12945487106075726 * context1 + -0.036980419717927253 * context2);
    float h7 = tanh(0.13258190863992203 + 0.56593199661556126 * context0 + -0.248383530839873 * context1 + 0.072026494433305083 * context2);
    float h8 = tanh(-0.017348567257133202 + -0.34536316394530059 * context0 + -0.068053535775343538 * context1 + 0.51978285271578328 * context2);
    float h9 = tanh(-0.1394279774471123 + -0.44925842041823805 * context0 + -0.41243852912861456 * context1 + 0.094268973437265396 * context2);
    float h10 = tanh(0.010126155971740358 + -0.021066112505267985 * context0 + -0.50187024796528268 * context1 + -0.43070869038965881 * context2);
    float h11 = tanh(0.4521273031819279 + -0.52883779187289326 * context0 + -0.55336457560123742 * context1 + -0.00049237338813473848 * context2);
    float raw0 = -1.8690579157914813 + -0.14549551881978479 * h0 + 0.45769496451769887 * h1 + 0.062354828705448014 * h2 + 0.14962199780478763 * h3 + -0.082780793993982124 * h4 + -0.2548567766569913 * h5 + 0.046094878721683837 * h6 + 0.1570668700454356 * h7 + 0.017233647351529699 * h8 + -0.21922113342637073 * h9 + -0.009417045926754717 * h10 + 0.31240044848145154 * h11;
    float raw1 = 0.053014931331505574 + -0.04154858507680554 * h0 + -0.026161415568588777 * h1 + 0.065573633389014813 * h2 + 0.039909009522261771 * h3 + -0.028568144968495301 * h4 + -0.00740022749797393 * h5 + 0.14634842064824474 * h6 + -0.020654185014871356 * h7 + 0.010011425576874166 * h8 + -0.010184681964805967 * h9 + 0.008987094121020861 * h10 + 0.092750449064018206 * h11;
    float raw2 = -1.3636830031947804 + -0.13656839706624269 * h0 + 0.42731708599433593 * h1 + 0.056865675422046506 * h2 + 0.15031747755804131 * h3 + -0.094086210064548523 * h4 + -0.2334865542643533 * h5 + 0.050465248608879416 * h6 + 0.19309503496652436 * h7 + 0.017730528531909165 * h8 + -0.23002370888958035 * h9 + -0.011033466568397697 * h10 + 0.3119923304753448 * h11;
    float diagonal0 = qualifiedSoftplus(raw0) + 1.0e-6;
    float offDiagonal = raw1;
    float diagonal1 = qualifiedSoftplus(raw2) + 1.0e-6;
    l00 = diagonal0 * diagonal0;
    l01 = diagonal0 * offDiagonal;
    l11 = offDiagonal * offDiagonal + diagonal1 * diagonal1;
}

void qualifiedMobility(
    int sample,
    ITensor<float, 2> inputContext,
    ITensor<float, 2> force,
    IWTensor<float, 2> output
)
{
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
}

void qualifiedCovariantMobility(
    int sample,
    ITensor<float, 2> inputContext,
    ITensor<float, 2> metric,
    ITensor<float, 2> baryonVelocity,
    ITensor<float, 2> carrierVelocity,
    ITensor<float, 2> force,
    IWTensor<float, 2> output
)
{
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
    {
        spatial /= spatialNorm;
    }
    else
    {
        float4 axis = float4(0.0, 1.0, 0.0, 0.0);
        float4 axisCovector = mul(g, axis);
        spatial = axis + u * dot(u, axisCovector);
        spatial /= sqrt(max(dot(spatial, mul(g, spatial)), 1.0e-18));
    }
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
}
