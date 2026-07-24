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
static const float T33Q_ENERGY_SCALE = 1.50987076e-08;
static const float T33Q_DRIFT_SCALE = 0.016;
static const float T33Q_PHASE_BASE = 0.00587578478;
static const float T33Q_PHASE_T = 0.000228064007;
static const float T33Q_PHASE_N = -1.19011276e-05;
static const float T33Q_PHASE_F = -5.09465317e-05;
static const float T33Q_DENSITY_MIN = 0.06;
static const float T33Q_DENSITY_MAX = 0.18;
static const float T33Q_TEMPERATURE_MIN =
    0.08;
static const float T33Q_TEMPERATURE_MAX =
    0.8;
static const float T33Q_FRACTION_MIN = 0.025;
static const float T33Q_FRACTION_MAX = 0.09;

static const int T33Q_NEGATIVE_LATTICE_COUNT = 24;
static const float T33Q_NEGATIVE_LATTICE[24] = { 0.8279356, 0.777134944, 0.726433031, 0.675830826, 0.625310258, 0.574850676, 0.524432295, 0.474035885, 0.423641833, 0.373229483, 0.322776768, 0.27226004, 0.221653955, 0.170931394, 0.120063434, 0.069019424, 0.0177672072, -0.0337266263, -0.0854969905, -0.1375822, -0.190029055, -0.242899599, -0.296261486, -0.350080581 };
static const int T33Q_POSITIVE_LATTICE_COUNT = 24;
static const float T33Q_POSITIVE_LATTICE[24] = { 0.521323174, 0.468914409, 0.416339622, 0.363485252, 0.310266377, 0.256617609, 0.202487188, 0.147832478, 0.0926172228, 0.0368092297, -0.019620983, -0.0767014614, -0.134459578, -0.192922693, -0.252118498, -0.312074884, -0.37281935, -0.434377058, -0.496767619, -0.559998588, -0.624055311, -0.688882964, -0.754358276, -0.820241809 };

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
        0.5 * 0.892004527 * y * y
        + 0.00117659738 * (qualifiedSoftplus(5 * (y - 0)) - 0.693147181 - 2.5 * y)
        + 0.00125993402 * (qualifiedSoftplus(5 * (y - 0.111111111)) - 0.45346348 - 1.8228822 * y)
        + 0.00141513187 * (qualifiedSoftplus(5 * (y - 0.222222222)) - 0.284571982 - 1.23831901 * y)
        + 0.00172678169 * (qualifiedSoftplus(5 * (y - 0.333333333)) - 0.173007989 - 0.794345524 * y)
        + 0.00243270284 * (qualifiedSoftplus(5 * (y - 0.444444444)) - 0.102888684 - 0.488862999 * y)
        + 0.00427972126 * (qualifiedSoftplus(5 * (y - 0.555555556)) - 0.0603201275 - 0.292684514 * y)
        + 0.00926524641 * (qualifiedSoftplus(5 * (y - 0.666666667)) - 0.0350524161 - 0.172225978 * y)
        + 0.0197211307 * (qualifiedSoftplus(5 * (y - 0.777777778)) - 0.0202614198 - 0.100287683 * y)
        + 0.0348923822 * (qualifiedSoftplus(5 * (y - 0.888888889)) - 0.0116752072 - 0.0580365822 * y)
        + 0.0517620529 * (qualifiedSoftplus(5 * (y - 1)) - 0.00671534849 - 0.0334642546 * y);
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
        / (-0.716666667 - (-1))
        - 1.0,
        -1.0,
        1.0
    );
    float positiveCoordinate = clamp(
        2.0 * (z - 0.616666667)
        / (1.05 - 0.616666667)
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
    float h0 = tanh(0.0592667157 + -0.243659371 * context0 + 0.475268933 * context1 + -0.117902925 * context2);
    float h1 = tanh(-0.264283815 + 0.118539255 * context0 + 0.107587614 * context1 + -0.0238120466 * context2);
    float h2 = tanh(-0.0176785437 + 0.0239421524 * context0 + -0.472113424 * context1 + -0.00317350568 * context2);
    float h3 = tanh(0.0340675792 + 0.336225594 * context0 + 0.328638996 * context1 + 0.213649789 * context2);
    float h4 = tanh(0.0798098004 + -0.0995863304 * context0 + 0.150370335 * context1 + 0.191256641 * context2);
    float h5 = tanh(-0.229450553 + -0.412819553 * context0 + 0.714957225 * context1 + 0.0640880183 * context2);
    float h6 = tanh(0.0559426466 + 0.332145144 * context0 + 0.129454871 * context1 + -0.0369804197 * context2);
    float h7 = tanh(0.132581909 + 0.565931997 * context0 + -0.248383531 * context1 + 0.0720264944 * context2);
    float h8 = tanh(-0.0173485673 + -0.345363164 * context0 + -0.0680535358 * context1 + 0.519782853 * context2);
    float h9 = tanh(-0.139427977 + -0.44925842 * context0 + -0.412438529 * context1 + 0.0942689734 * context2);
    float h10 = tanh(0.010126156 + -0.0210661125 * context0 + -0.501870248 * context1 + -0.43070869 * context2);
    float h11 = tanh(0.452127303 + -0.528837792 * context0 + -0.553364576 * context1 + -0.000492373388 * context2);
    float raw0 = -1.86905792 + -0.145495519 * h0 + 0.457694965 * h1 + 0.0623548287 * h2 + 0.149621998 * h3 + -0.082780794 * h4 + -0.254856777 * h5 + 0.0460948787 * h6 + 0.15706687 * h7 + 0.0172336474 * h8 + -0.219221133 * h9 + -0.00941704593 * h10 + 0.312400448 * h11;
    float raw1 = 0.0530149313 + -0.0415485851 * h0 + -0.0261614156 * h1 + 0.0655736334 * h2 + 0.0399090095 * h3 + -0.028568145 * h4 + -0.0074002275 * h5 + 0.146348421 * h6 + -0.020654185 * h7 + 0.0100114256 * h8 + -0.010184682 * h9 + 0.00898709412 * h10 + 0.0927504491 * h11;
    float raw2 = -1.363683 + -0.136568397 * h0 + 0.427317086 * h1 + 0.0568656754 * h2 + 0.150317478 * h3 + -0.0940862101 * h4 + -0.233486554 * h5 + 0.0504652486 * h6 + 0.193095035 * h7 + 0.0177305285 * h8 + -0.230023709 * h9 + -0.0110334666 * h10 + 0.31199233 * h11;
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
