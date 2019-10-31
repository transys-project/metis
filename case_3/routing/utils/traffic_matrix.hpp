
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

typedef std::vector<std::vector<double>> Matrix2D;

class TrafficMatrix
{

public:
    TrafficMatrix() {}
    TrafficMatrix(int seed)
    {
        this->seed = seed;
        this->rng.seed(this->seed);
        this->_generated = false;
    }

    ~TrafficMatrix() {}

    void generateRandomTrafficMatrix(int size, int trafficFactor)
    {
        std::cout << "size is" << size << std::endl;
        for (int i = 0; i < size; i++)
        {
            this->trafficMatrix.push_back(std::vector<double>());
            for (int j = 0; j < size; j++)
            {
                if (j == i)
                {
                    this->trafficMatrix[i].push_back(0);
                }
                else {
                    double trafficDensity = this->uniformGenerator(this->rng) * 0.9 + 0.1;
                    this->trafficMatrix[i].push_back(trafficDensity * trafficFactor / (size - 1));
                }
            }
        }
        this->_generated = true;
    }

    const Matrix2D &getTrafficMatrix() const
    {
        if (this->_generated == false)
        {
            std::cout << "ERROR! Traffic Matrix has not been generated." << std::endl;
        }
        return this->trafficMatrix;
    }

    void serializeTrafficMatrix(const char *filename) const
    {

        if (this->_generated == false)
        {
            std::cout << "ERROR! Traffic Matrix has not been generated." << std::endl;
        }
        std::ofstream FILE;
        FILE.open(filename);
        for (auto &row : this->trafficMatrix)
        {
            for (auto &value : row)
            {
                FILE << value << " ";
            }
            FILE << std::endl;
        }
        FILE.close();
    }

private:
    Matrix2D trafficMatrix;
    int seed;
    boost::random::uniform_01<double> uniformGenerator;
    boost::random::mt19937 rng;
    bool _generated;
};
