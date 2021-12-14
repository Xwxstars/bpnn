#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>

#define INNODE 2 //输入层节点个数
#define HIDENODE 4  //隐藏层节点个数
#define OUTNODE 1   //输出层节点个数

double rate = 0.8;  //步长
double threshold = 1e-3;    //允许误差
size_t mosttimes = 1e6; //最大迭代次数

//采集的样本数据，并存储
struct Sample {
    std::vector<double> in, out;
};

//构造节点数据
struct Node {
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight, weight_delta;
};


//制作需要的工具函数包
namespace utils {

    //激活函数
    inline double sigmoid(double x) {
        double res = 1.0 / (1.0 + std::exp(-x));
        return res;
    }
    //读取文件的数据的函数，并存在一个vector中
    std::vector<double> getFileData(std::string filename) {
        std::vector<double> res;

        std::ifstream in(filename);
        if (in.is_open()) {
            while (!in.eof()) {
                double buffer;
                in >> buffer;
                res.push_back(buffer);
            }
            in.close();
        } else {
            std::cout <<"Error in reading"<< filename << std::endl;
        }

        return res;
    }
    //获取训练数据，并存储在一个样本vector中
    std::vector<Sample> getTrainData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE + OUTNODE) {
            Sample tmp;
            for (size_t t = 0; t < INNODE; t++) {
                tmp.in.push_back(buffer[i + t]);
            }
            for (size_t t = 0; t < OUTNODE; t++) {
                tmp.out.push_back(buffer[i + INNODE + t]);
            }
            res.push_back(tmp);
        }

        return res;
    }
    //获取测试数据，存储在一个样本vector中
    std::vector<Sample> getTestData(std::string filename) {
        std::vector<Sample> res;

        std::vector<double> buffer = getFileData(filename);

        for (size_t i = 0; i < buffer.size(); i += INNODE) {
            Sample tmp;
            for (size_t t = 0; t < INNODE; t++) {
                tmp.in.push_back(buffer[i + t]);
            }
            res.push_back(tmp);
        }

        return res;
    }

}
//创建输入层、输出层和隐藏层的节点数组
Node *inputLayer[INNODE], *hideLayer[HIDENODE], *outLayer[OUTNODE];

inline void init() {
    //创建随机数种子
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);
    //初始化输入层到隐藏层的权值和权值变化值
    for (size_t i = 0; i < INNODE; i++) {
        //遍历所有的输入层节点，在堆创建。
        ::inputLayer[i] = new Node();
        //每个输入层到隐藏层都有权值
        for (size_t j = 0; j < HIDENODE; j++) {
            ::inputLayer[i]->weight.push_back(distribution(rd));
            ::inputLayer[i]->weight_delta.push_back(0.f);
        }
    }
    //初始化隐藏层到输出层的权值二号权值变化值
    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i] = new Node();
        ::hideLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < OUTNODE; j++) {
            ::hideLayer[i]->weight.push_back(distribution(rd));
            ::hideLayer[i]->weight_delta.push_back(0.f);
        }
    }
    //初始化输出层的偏置值
    for (size_t i = 0; i < OUTNODE; i++) {
        ::outLayer[i] = new Node();
        ::outLayer[i]->bias = distribution(rd);
    }

}
//复位偏置值变化量
inline void reset_delta() {

    for (size_t i = 0; i < INNODE; i++) {
        ::inputLayer[i]->weight_delta.assign(::inputLayer[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < HIDENODE; i++) {
        ::hideLayer[i]->bias_delta = 0.f;
        ::hideLayer[i]->weight_delta.assign(::hideLayer[i]->weight_delta.size(), 0.f);
    }

    for (size_t i = 0; i < OUTNODE; i++) {
        ::outLayer[i]->bias_delta = 0.f;
    }

}

int main(int argc, char *argv[]) {
    
    init();
    //获得训练需要的数据，存在sample的vector中
    std::vector<Sample> train_data = utils::getTrainData("traindata.txt");

    // training
    for (size_t times = 0; times < mosttimes; times++) {
        //每次训练都需要将变化偏置置零
        reset_delta();
        //定义最大误差的变量
        double error_max = 0.f;
        //以sample为单元，遍历所有的训练数据
        for (auto idx : train_data) {
            //将训练数据输入到输入层节点的value中
            for (size_t i = 0; i < INNODE; i++) {
                ::inputLayer[i]->value = idx.in[i];
            }

            // 正向传播
            for (size_t j = 0; j < HIDENODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < INNODE; i++) {
                    sum += ::inputLayer[i]->value * ::inputLayer[i]->weight[j];
                }
                sum -= ::hideLayer[j]->bias;

                ::hideLayer[j]->value = utils::sigmoid(sum);
            }
            //计算得出输出层的value
            for (size_t j = 0; j < OUTNODE; j++) {
                double sum = 0;
                for (size_t i = 0; i < HIDENODE; i++) {
                    sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
                }
                sum -= ::outLayer[j]->bias;

                ::outLayer[j]->value = utils::sigmoid(sum);
            }

            // 计算误差
            double error = 0.f;
            for (size_t i = 0; i < OUTNODE; i++) {
                double tmp = std::fabs(::outLayer[i]->value - idx.out[i]);
                error += tmp * tmp / 2;
            }

            error_max = error;
            
        
        

            // 反向传播

            for (size_t i = 0; i < OUTNODE; i++) {
                double bias_delta = -(idx.out[i] - ::outLayer[i]->value) *
                        ::outLayer[i]->value * (1.0 - ::outLayer[i]->value);
                ::outLayer[i]->bias_delta += bias_delta;
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                for (size_t j = 0; j < OUTNODE; j++) {
                    double weight_delta = (idx.out[j] - ::outLayer[j]->value) *
                            ::outLayer[j]->value * (1.0 - ::outLayer[j]->value) *
                            ::hideLayer[i]->value;
                    ::hideLayer[i]->weight_delta[j] += weight_delta;
                }
            }

            for (size_t i = 0; i < HIDENODE; i++) {
                double sum = 0;
                for (size_t j = 0; j < OUTNODE; j++) {
                    sum += -(idx.out[j] - ::outLayer[j]->value) *
                            ::outLayer[j]->value * (1.0 - ::outLayer[j]->value) *
                            ::hideLayer[i]->weight[j];
                }
                ::hideLayer[i]->bias_delta +=
                        sum * ::hideLayer[i]->value * (1.0 - ::hideLayer[i]->value);
            }

            for (size_t i = 0; i < INNODE; i++) {
                for (size_t j = 0; j < HIDENODE; j++) {
                    double sum = 0.f;
                    for (size_t k = 0; k < OUTNODE; k++) {
                        sum += (idx.out[k] - ::outLayer[k]->value) *
                                ::outLayer[k]->value * (1.0 - ::outLayer[k]->value) *
                                ::hideLayer[j]->weight[k];
                    }
                    ::inputLayer[i]->weight_delta[j] +=
                            sum *
                            ::hideLayer[j]->value * (1.0 - ::hideLayer[j]->value) *
                            ::inputLayer[i]->value;
                }
            }

        }
        if (error_max <= threshold) {
            std::cout << "Success with " << times + 1 << " times training." << std::endl;
            std::cout << "Maximum error: " << error_max << std::endl;
            break;
        }

        auto train_data_size = double(train_data.size());

        for (size_t i = 0; i < INNODE; i++) {
            for (size_t j = 0; j < HIDENODE; j++) {
                ::inputLayer[i]->weight[j] +=
                        rate * ::inputLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < HIDENODE; i++) {
            ::hideLayer[i]->bias +=
                    rate * ::hideLayer[i]->bias_delta / train_data_size;
            for (size_t j = 0; j < OUTNODE; j++) {
                ::hideLayer[i]->weight[j] +=
                        rate * ::hideLayer[i]->weight_delta[j] / train_data_size;
            }
        }

        for (size_t i = 0; i < OUTNODE; i++) {
            ::outLayer[i]->bias +=
                    rate * ::outLayer[i]->bias_delta / train_data_size;
        }

    }

    std::vector<Sample> test_data = utils::getTestData("testdata.txt");

    // predict
    for (auto &idx : test_data) {

        for (size_t i = 0; i < INNODE; i++) {
            ::inputLayer[i]->value = idx.in[i];
        }

        for (size_t j = 0; j < HIDENODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < INNODE; i++) {
                sum += ::inputLayer[i]->value * inputLayer[i]->weight[j];
            }
            sum -= ::hideLayer[j]->bias;

            ::hideLayer[j]->value = utils::sigmoid(sum);
        }

        for (size_t j = 0; j < OUTNODE; j++) {
            double sum = 0;
            for (size_t i = 0; i < HIDENODE; i++) {
                sum += ::hideLayer[i]->value * ::hideLayer[i]->weight[j];
            }
            sum -= ::outLayer[j]->bias;

            ::outLayer[j]->value = utils::sigmoid(sum);

            idx.out.push_back(::outLayer[j]->value);

            for (auto &tmp : idx.in) {
                std::cout << tmp << " ";
            }
            for (auto &tmp : idx.out) {
                std::cout << tmp << " ";
            }
            std::cout << std::endl;
        }

    }
    for(int i=0;i<INNODE;i++)
    {
        std::cout<<inputLayer[i]->weight[1]<<"  "<<inputLayer[i]->weight[2]<<std::endl;
        std::cout<<sizeof(inputLayer[i]->weight)<<std::endl;
    }

    return 0;
}
