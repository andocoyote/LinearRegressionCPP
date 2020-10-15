#include "AndoRegression.h"

AndoRegression::AndoRegression()
{
    ;
}

AndoRegression::AndoRegression(string FileName, string CostFile)
{
    this->filename = FileName;
    this->costfile = CostFile;

    isConstructed = ProcessDataFile();
}

AndoRegression::~AndoRegression()
{
    if (matrix) delete matrix;
    if (thetas) delete thetas;
    if (features) delete features;
    if (labels) delete labels;
}

bool AndoRegression::ContainsAlpha(string str)
{
    for (char const& ch : str) {
        if ((ch <= 'z' && ch >= 'a') || (ch <= 'Z' && ch >= 'A'))
        {
            return true;
        }
    }

    return false;
}

bool AndoRegression::ProcessDataFile()
{
    string line = "";
    list<double> tokens;
    bool success = true;

    ifstream datafile(filename);

    if (datafile.is_open())
    {
        size_t rowcount = 0;
        size_t colcount = 0;
        size_t startpos = 0;
        size_t endpos = 0;
        string token = "";

        // In order to create the matrix, we need to know how many rows are in the file
        // and how many tokens are in each row
        while (getline(datafile, line))
        {
            if (ContainsAlpha(line))
            {
                cout << "\"" << line << "\" is probably the columns header. Skipping." << endl;
                continue;
            }

            startpos = 0;
            endpos = 0;

            while (endpos < line.length())
            {
                // Find the first alpha - numberic char to trim whitespace from the head (not tab, space, or comma)
                startpos = line.find_first_not_of("\x9\x20\x2C", endpos);

                // Get the string after deleting leading whitespace
                line = line.substr(startpos, line.length());

                // Get the end of the first alpha - numberic token
                endpos = line.find_first_of("\x9\x20\x2C", 0);

                // Now, startpos is the first char and endpos is the last char in the token
                endpos = endpos < line.length() ? endpos : line.length();
                token = line.substr(0, endpos);

                tokens.push_back(strtod(token.c_str(), NULL));
            }

            rowcount++;
        }

        datafile.close();

        colcount = tokens.size() / rowcount;

        matrix = new(nothrow) MatrixXd(rowcount, colcount);
        thetas = new(nothrow) VectorXd(2);

        // Fill the matrix with the tokens parsed from the data file
        for (int i = 0; i < matrix->rows(); i++)
        {
            for (int j = 0; j < matrix->cols(); j++)
            {
                (*matrix)(i, j) = tokens.front();
                tokens.pop_front();
            }
        }

        // Create a matrix of features where the first column is ones
        features = new(nothrow) MatrixXd(rowcount, 2);
        features->col(0) = VectorXf::Ones(rowcount).cast<double>();
        features->col(1) = matrix->col(0);

        labels = new(nothrow) VectorXd(matrix->col(1));

        matrixSize = std::make_tuple(matrix->rows(), matrix->cols());
        featureSize = std::make_tuple(features->rows(), features->cols());
        labelSize = std::make_tuple(labels->rows(), labels->cols());
    }
    else
    {
        success = false;
        goto cleanup;
    }

cleanup:

    return success;
}

void AndoRegression::Normalize(MatrixXd** norm_features, VectorXd** norm_labels)
{
    *norm_features = new MatrixXd(*features);
    *norm_labels = new VectorXd(*labels);

    (*norm_features)->col(1).normalize();
    (*norm_labels)->normalize();
}

void AndoRegression::Regress(MatrixXd* X, VectorXd* y, double alpha, double epsilon, int epochs)
{
    ofstream costfile;
    double prev_cost = 0;
    cost = 0;
    
    costfile.open(this->costfile, ios::out | ios::trunc);

    srand(42);

    // Initialize our thetas to random
    (*thetas)(0) = ((double)rand() / (RAND_MAX)) + 1;
    (*thetas)(1) = ((double)rand() / (RAND_MAX)) + 1;

    cout << "Initial thetas: " << (*thetas)(0) << " " << (*thetas)(1) << endl;

    // Calculate parameters theta and cost
    for (int i = 0; i < epochs; i++)
    {
        VectorXd yhat = (*X) * (*thetas);
        VectorXd error = yhat - *y;
        VectorXd error_squared = error.unaryExpr([](double d) {return std::pow(d, 2); });

        // Compute the cost
        prev_cost = cost;
        cost = error_squared.mean();

        // Compute our new thetas
        (*thetas) = (*thetas) - (alpha * ((*X).transpose() * error) / (*X).rows());

        cout << std::setprecision(12) << "Cost: " << cost << " Previous Cost: " << prev_cost << endl;

        // Write every 1000th cost and thetas to a CSV file for graphing
        if (i % 1000 == 0 && costfile.is_open())
        {
            costfile << (*thetas)(0) << "," << (*thetas)(1) << "," << fabs(cost) << endl;
        }

        // Exit when the change in cost is < threshold
        if ( fabs(cost - prev_cost) < epsilon)
        {
            cout << "The algorithm hit epsilon at " << i << " epochs." << endl;
            cout << "Minimized cost to " << cost << endl;
            cout << "Theta0 which minimizes the cost: " << (*thetas)(0) << endl;
            cout << "Theta1 which minimizes the cost: " << (*thetas)(1) << endl;
            break;
        }
    }

    costfile.close();
}