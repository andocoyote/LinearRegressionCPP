#include <string>
#include <stdio.h>
#include <Eigen/Dense>
#include "AndoRegression.h"

using namespace Eigen;
using namespace std;


int main()
{
    // For running with the original data
    MatrixXd* features = nullptr;
    VectorXd* labels = nullptr;

    // For running with the normalized data
    MatrixXd* norm_features = nullptr;
    VectorXd* norm_labels = nullptr;

    string filename = "C:\\Users\\amikesel\\OneDrive\\Engineering\\MachineLearning\\Projects\\pytorch_test\\education_vs_poverty_log10.csv";
    string costfile = "C:\\users\\amikesel\\Desktop\\cppdata.csv";
    AndoRegression* andoRegression = new(nothrow) AndoRegression(filename, costfile);

    // Dump a bunch of properties to make sure we're initialzed correctly and the data looks right
    string success_result = andoRegression->IsConstructed() ? "successfully" : "unsuccessfully";

    cout << "AndoRegression was " << success_result << " constructed." << endl;
    cout << "AndoRegression data matrix size is " << std::get<0>(andoRegression->MatrixSize()) << " x " << std::get<1>(andoRegression->MatrixSize()) << endl;
    cout << "AndoRegression feature matrix size is " << std::get<0>(andoRegression->FeatureSize()) << " x " << std::get<1>(andoRegression->FeatureSize()) << endl;
    cout << "AndoRegression label vector size is " << std::get<0>(andoRegression->LabelSize()) << " x " << std::get<1>(andoRegression->LabelSize()) << endl;

    features = andoRegression->Features();
    labels = andoRegression->Labels();

    // Run with the original data
    cout << "Features:\n" << *features << endl;
    cout << "Values:\n" << *labels << endl;

    cout << "Running linear regression with original data ..." << endl;

    // alpha, epsilon, epochs: 0.00001, 0.000000000001, 100000000
    andoRegression->Regress(features, labels, 0.01, 0.000000001, 10000000);

    // Release all of our resources
    if (andoRegression && andoRegression->IsConstructed())
    {
        delete andoRegression;
    }

    if (norm_features)
    {
        delete norm_features;
    }

    if (norm_labels)
    {
        delete norm_labels;
    }
}