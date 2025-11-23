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
            Perceptron andPerceptron = new Perceptron(andInputs[0].Length, 0.1, new Random(), mseFunction);
            double error = 1;
            do
            {
                error = andPerceptron.TrainWithHillClimbing(andInputs, new double[] { 0, 1, 1, 1 }, error);
                Console.WriteLine(error);
                Console.WriteLine(Math.Round(andPerceptron.Compute([0, 0])));
                Console.WriteLine(Math.Round(andPerceptron.Compute([1, 0])));
                Console.WriteLine(Math.Round(andPerceptron.Compute([0, 1])));
                Console.WriteLine(Math.Round(andPerceptron.Compute([1, 1])));
            } while (error > 0.1);

        }

       

    }
}
