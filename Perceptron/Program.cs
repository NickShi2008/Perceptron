using System.Reflection.Metadata.Ecma335;

namespace Perceptron
{

    internal class Program
    {
        static void Main(string[] args)
        {
            //double[] initialWeights = {0, 1};
            //double bias = 1;
            Func<double, double, double> mseFunction = (expected, input) =>
            {
                return Math.Pow(expected - input, 2);
            };  
            double[][] andInputs = {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 },
            };
            var sigmoid = new ActivationFunction(
            x => 1.0 / (1.0 + Math.Exp(-x)),
            x => {
                double y = 1.0 / (1.0 + Math.Exp(-x));
                return y * (1 - y);
            });
            ErrorFunction errorFunction = new ErrorFunction(mseFunction, (input, expected) =>
            {
                return (input - expected);
            });
            Perceptron andPerceptron = new Perceptron(andInputs[0].Length, 0.1, sigmoid, errorFunction);
            double error = 1;
            do
            {
                error = andPerceptron.Train(andInputs, new double[] { 0, 0, 0, 1 });
                Console.WriteLine(error);
                Console.WriteLine(andPerceptron.Compute([0, 0]));
                Console.WriteLine(andPerceptron.Compute([1, 0]));
                Console.WriteLine(andPerceptron.Compute([0, 1]));
                Console.WriteLine(andPerceptron.Compute([1, 1]));
            } while (error > 0.1);

        }

       

    }
}
