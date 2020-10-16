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
    if (data) delete data;
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

        // Instantiate matrix to hold all of the data read from the file
        data = new(nothrow) MatrixXd(rowcount, colcount);
        if (!data)
        {
            cout << "Error: Failed to instantiate matrix data matrix." << endl;
            success = false;
            goto cleanup;
        }

        // Intantiate vector to hold our thetas
        thetas = new(nothrow) VectorXd(2);
        if (!thetas)
        {
            cout << "Error: Failed to instantiate thetas vector." << endl;
            success = false;
            goto cleanup;
        }

        // Fill the matrix with the tokens parsed from the data file
        for (int i = 0; i < data->rows(); i++)
        {
            for (int j = 0; j < data->cols(); j++)
            {
                (*data)(i, j) = tokens.front();
                tokens.pop_front();
            }
        }

        // Create a matrix of features where the first column is ones
        features = new(nothrow) MatrixXd(rowcount, 2);
        if (!features)
        {
            cout << "Error: Failed to instantiate features matrix." << endl;
            success = false;
            goto cleanup;
        }

        features->col(0) = VectorXf::Ones(rowcount).cast<double>();
        features->col(1) = data->col(0);

        labels = new(nothrow) VectorXd(data->col(1));
        if (!labels)
        {
            cout << "Error: Failed to instantiate labels vector." << endl;
            success = false;
            goto cleanup;
        }

        // Create the tuples which hold the sizes of our respective matrices and vectors
        matrixSize = std::make_tuple((int)data->rows(), (int)data->cols());
        featureSize = std::make_tuple((int)features->rows(), (int)features->cols());
        labelSize = std::make_tuple((int)labels->rows(), (int)labels->cols());
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

    VectorXd* yhat = nullptr;
    VectorXd* error = nullptr;
    VectorXd* error_squared = nullptr;

    // Calculate parameters theta and cost
    for (int i = 0; i < epochs; i++)
    {
        // Compute yhat (our predictions)
        yhat = new VectorXd((*X) * (*thetas));
        if (!yhat)
        {
            cout << "Error: Failed to instantiate yhat vector." << endl;
            goto cleanup;
        }

        // Compute the error
        error = new VectorXd(*yhat - *y);
        if (!error)
        {
            cout << "Error: Failed to instantiate error vector." << endl;
            goto cleanup;
        }

        // Compute error squared
        error_squared = new VectorXd(error->unaryExpr([](double d) {return std::pow(d, 2); }));
        if (!error_squared)
        {
            cout << "Error: Failed to instantiate error_squared vector." << endl;
            goto cleanup;
        }

        // Compute the cost/loss
        prev_cost = cost;
        cost = error_squared->mean();

        // Our algorithm in pseudo-code: the average of our gradients * learning rate
        //   thetas = thetas - alpha * ((X' * (X*thetas-y)) / m);
        (*thetas) = (*thetas) - (alpha * ((*X).transpose() * *error) / (*X).rows());

        cout << std::setprecision(12) << "Cost: " << cost << " Previous Cost: " << prev_cost << endl;

        // Write every nth cost and thetas to a CSV file for graphing
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

        if (yhat)
        {
            delete yhat;
            yhat = nullptr;
        }

        if (error)
        {
            delete error;
            error = nullptr;
        }

        if (error_squared)
        {
            delete error_squared;
            error_squared = nullptr;
        }
    }

cleanup:

    if (yhat) delete yhat;
    if (error) delete error;
    if (error_squared) delete error_squared;

    costfile.close();
}