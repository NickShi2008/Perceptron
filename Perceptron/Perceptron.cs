namespace Perceptron
{
    public class Perceptron
    {
        public double[] weights;
        double bias;
        ErrorFunction errorFunc;
        ActivationFunction activationFunction;
        public double LearningRate { get; set; }


        public Perceptron(int amountOfInputs, double learningRate,
        ActivationFunction activationFunction, ErrorFunction errorFunction)
        {
            this.weights = new double[amountOfInputs];
            this.LearningRate = learningRate;
            this.activationFunction = activationFunction;
            this.errorFunc = errorFunction;
        }

        public void Randomize(Random random, double min, double max)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = random.NextDouble() * (max - min) + min;
            }
            bias = random.NextDouble() * (max - min) + min;
        }

        public double Compute(double[] inputs)
        {
            double val = bias;
            for (int i = 0; i < inputs.Length; i++)
            {
                val += inputs[i] * weights[i];
            }

            return activationFunction.Function(val);
        }
        public double ComputeRaw(double[] inputs)
        {
            double val = bias;
            for (int i = 0; i < inputs.Length; i++)
            {
                val += inputs[i] * weights[i];
            }
            return val;
        }

        public double[] Compute(double[][] inputs)
        {
            double[] outputs = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = Compute(inputs[i]);
            }
            return outputs;
        }

        public double GetError(double[][] inputs, double[] desiredOutputs)
        {
            double totalError = 0.0;
            double[] outputs = Compute(inputs);
            for (int i = 0; i < outputs.Length; i++)
            {
                totalError += errorFunc.Function(desiredOutputs[i], outputs[i]);
            }
            return totalError / desiredOutputs.Length;
        }

        public double Train(double[] inputs, double desiredOutput)
        { 
            double changeInBias = 0.0;
            double[] changeInWeights = new double[weights.Length];

            double biasPartialDerivative = activationFunction.Derivative(ComputeRaw(inputs)) * errorFunc.Derivative(Compute(inputs), desiredOutput);
            changeInBias = -LearningRate * biasPartialDerivative;
            for (int i = 0; i < weights.Length; i++)
            {
                double weightPartialDerivative = 
                    biasPartialDerivative * activationFunction.Derivative(bias + ComputeRaw(inputs)) * weights[i];
                changeInWeights[i] = -LearningRate * weightPartialDerivative;

                weights[i] += changeInWeights[i];
            }
            bias += changeInBias;
            return Math.Abs(errorFunc.Function(Compute(inputs), desiredOutput));
        }

        public double Train(double[][] inputs, double[] desiredOutput)
        { 
            double totalError = 0.0;
            for(int i = 0 ; i < inputs.Length; i++)
            {
                totalError += Train(inputs[i], desiredOutput[i]);
            }
            return totalError / inputs.Length;
        }
    }
}
