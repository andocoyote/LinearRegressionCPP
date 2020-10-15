#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <list>
#include <tuple>
#include <Eigen/Dense>
#include <math.h>
#include <stdlib.h>

using namespace Eigen;
using namespace std;

class AndoRegression
{
public:
    AndoRegression();
    AndoRegression(string FileName, string CostFile);

    ~AndoRegression();

    bool IsConstructed() { return isConstructed; }

    MatrixXd* Features() { return features; }
    VectorXd* Labels() { return labels; }
    VectorXd* Coefficients() { return thetas; }

    tuple<int, int> MatrixSize() { return matrixSize; }
    tuple<int, int> FeatureSize() { return featureSize; }
    tuple<int, int> LabelSize() { return labelSize; }
    double Cost() { return cost; }

    void Normalize(MatrixXd** norm_features, VectorXd** norm_labels);
    void Regress(MatrixXd* X, VectorXd* y, double alpha=0.000001, double epsilon=0.00001, int epochs=10000000);

private:
    bool ProcessDataFile();
    bool ContainsAlpha(string str);

    string filename = "";
    string costfile = "";
    bool isConstructed = false;
    MatrixXd* matrix = nullptr;
    VectorXd* thetas = nullptr;
    MatrixXd* features = nullptr;
    VectorXd* labels = nullptr;
    tuple<int, int> matrixSize;
    tuple<int, int> featureSize;
    tuple<int, int> labelSize;
    double cost = 0;
};