namespace Perceptron
{
    public class Perceptron
    {
        public double[] weights;
        double bias;
        double mutationAmount;
        Random random;
        Func<double, double, double> errorFunc;

        public Perceptron(double[] initialWeightValues, double initialBiasValue,
        double mutationAmount, Random random, Func<double, double, double> errorFunc)
        {
            weights = initialWeightValues;
            bias = initialBiasValue;
            this.mutationAmount = mutationAmount;
            this.random = random;
            this.errorFunc = errorFunc;
        }

        public Perceptron(int amountOfInputs, double mutationAmount, Random random,
        Func<double, double, double> errorFunc)
        {
            weights = new double[amountOfInputs];
            Randomize(random, 0, mutationAmount);
            bias = 0.0;
            this.mutationAmount = mutationAmount;
            this.random = random;
            this.errorFunc = errorFunc;
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
                totalError += errorFunc(desiredOutputs[i], outputs[i]);
            }
            return totalError / desiredOutputs.Length;
        }

        public double TrainWithHillClimbing(double[][] inputs, double[] desiredOutputs, double currentError)
        {
            double initialError = currentError;
            int randIndex = (int)random.NextDouble() * weights.Length;
            double randomMutation = (random.NextDouble() * 2 - 1) * mutationAmount;
            bool biasOrWeight = random.Next(1, 3) % 2 == 0;
            if (biasOrWeight)
            {
                weights[randIndex] += randomMutation;
            }
            else
            {
                bias += randomMutation;
            }

            double newError = GetError(inputs, desiredOutputs);

            if (newError < initialError)
            {
                return newError;
            }
            else
            {
                if (biasOrWeight)
                    weights[randIndex] -= randomMutation;
                else
                    bias -= randomMutation;
            }
            return initialError;
        }
    }
}
