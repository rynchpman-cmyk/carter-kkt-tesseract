import "slangpy";

static const int MASTER_WIDTH = 32;
static const int MW_SKIP_W = 0;
static const int MW_SKIP_B = 5;
static const int MW_L1_W = 6;
static const int MW_L1_B = 166;
static const int MW_L2_W = 198;
static const int MW_L2_B = 1222;
static const int MW_L3_W = 1254;
static const int MW_L3_B = 1286;

// Native Vulkan reverse-mode kernel for the learned Carter scalar. Weight
// loads participate in Slang AD and accumulate into one shared gradient tensor.
[Differentiable]
void carterMaster(
    int sample,
    IDiffTensor<float, 2> features,
    IDiffTensor<float, 1> weights,
    IWDiffTensor<float, 1> output)
{
    float hidden1[MASTER_WIDTH] = {};
    float hidden2[MASTER_WIDTH] = {};
    float value = weights[MW_SKIP_B];

    [ForceUnroll]
    for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
        value += weights[MW_SKIP_W + inputIndex] * features[sample, inputIndex];

    [ForceUnroll]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        float preactivation = weights[MW_L1_B + neuron];
        [ForceUnroll]
        for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
        {
            preactivation +=
                weights[MW_L1_W + neuron * 5 + inputIndex]
                * features[sample, inputIndex];
        }
        hidden1[neuron] = tanh(preactivation);
    }

    [ForceUnroll]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        float preactivation = weights[MW_L2_B + neuron];
        [ForceUnroll]
        for (int inputIndex = 0; inputIndex < MASTER_WIDTH; ++inputIndex)
        {
            preactivation +=
                weights[MW_L2_W + neuron * MASTER_WIDTH + inputIndex]
                * hidden1[inputIndex];
        }
        hidden2[neuron] = tanh(preactivation);
    }

    value += weights[MW_L3_B];
    [ForceUnroll]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
        value += weights[MW_L3_W + neuron] * hidden2[neuron];

    output[sample] = value;
}

[Differentiable]
float evalMasterValues(
    float feature0,
    float feature1,
    float feature2,
    float feature3,
    float feature4,
    IDiffTensor<float, 1> weights)
{
    float features[5] = {
        feature0, feature1, feature2, feature3, feature4
    };
    float hidden1[MASTER_WIDTH] = {};
    float hidden2[MASTER_WIDTH] = {};
    float value = weights[MW_SKIP_B];

    [ForceUnroll]
    for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
        value += weights[MW_SKIP_W + inputIndex] * features[inputIndex];

    [ForceUnroll]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        float preactivation = weights[MW_L1_B + neuron];
        [ForceUnroll]
        for (int inputIndex = 0; inputIndex < 5; ++inputIndex)
        {
            preactivation +=
                weights[MW_L1_W + neuron * 5 + inputIndex]
                * features[inputIndex];
        }
        hidden1[neuron] = tanh(preactivation);
    }

    [ForceUnroll]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
    {
        float preactivation = weights[MW_L2_B + neuron];
        [ForceUnroll]
        for (int inputIndex = 0; inputIndex < MASTER_WIDTH; ++inputIndex)
        {
            preactivation +=
                weights[MW_L2_W + neuron * MASTER_WIDTH + inputIndex]
                * hidden1[inputIndex];
        }
        hidden2[neuron] = tanh(preactivation);
    }

    value += weights[MW_L3_B];
    [ForceUnroll]
    for (int neuron = 0; neuron < MASTER_WIDTH; ++neuron)
        value += weights[MW_L3_W + neuron] * hidden2[neuron];
    return value;
}

// A bounded recurrence used to validate reverse-through-time generation and
// the shared-weight gradient accumulation ABI before the complete Carter/KKT
// step is migrated from GLSL into this differentiable Slang module.
[Differentiable]
void carterBpttProbe(
    int sample,
    IDiffTensor<float, 1> initialZ,
    IDiffTensor<float, 1> weights,
    IWDiffTensor<float, 1> output)
{
    float z = initialZ[sample];
    [ForceUnroll]
    for (int step = 0; step < 2; ++step)
    {
        float lambda = evalMasterValues(
            z,
            z * z,
            sin(z),
            cos(z),
            tanh(z),
            weights
        );
        z = 0.85 * z + 0.15 * tanh(lambda);
    }
    output[sample] = z;
}
