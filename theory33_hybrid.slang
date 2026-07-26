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
