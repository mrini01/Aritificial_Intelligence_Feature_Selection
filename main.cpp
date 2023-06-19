#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <numeric>
#include <sstream>
#include <string>
#include <iomanip>
#include <map>
#include <vector>

using namespace std;

//defining the data we have
struct Data {
  int classLabel;
  vector<double> features;
};

// the NN classifier we need, starting from part 2
class Classifier {
public:
  vector<Data> trainingInstances;

  void Train(const vector<Data> &trainingData) {
    trainingInstances = trainingData;
  }

  int Test(const Data &testData) {
    double minDistance = std::numeric_limits<double>::max();
    int nearestNeighborClass = -1;

    for (const auto &instance : trainingInstances) {
      double currentDistance =
          EuclideanDistance(instance.features, testData.features);

      if (currentDistance < minDistance) {
        minDistance = currentDistance;
        nearestNeighborClass = instance.classLabel;
      }
    }

    return nearestNeighborClass;
  }

  double EuclideanDistance(const vector<double> &v1, const vector<double> &v2) {
    if (v1.size() != v2.size()) {
      cout << "Error: vectors are of different sizes!" << endl;
      return -1;
    }

    double sum = 0;
    for (size_t i = 0; i < v1.size(); i++) {
      sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum);
  }
};

//our the data file is read
vector<Data> loadData(const string &filename) {
  vector<Data> data;
  ifstream file(filename);

  if (!file) {
    cerr << "Unable to open file: " << filename << endl;
    exit(1);
  }

  string line;
  while (getline(file, line)) {
    istringstream iss(line);
    Data d;
    iss >> d.classLabel;
    double feature;
    while (iss >> feature) {
      d.features.push_back(feature);
    }

    data.push_back(d);
  }

  file.close();

  return data;
}

//selecting the needed features
vector<Data> SelectFeatures(const vector<Data> &data,const vector<int> &selectedFeaturesIndices) {
  vector<Data> dataWithSelectedFeatures;

  for (auto &d : data) {
    Data newData;
    newData.classLabel = d.classLabel;
    for (auto &index : selectedFeaturesIndices) {
      newData.features.push_back(d.features[index]);
    }
    dataWithSelectedFeatures.push_back(newData);
  }

  return dataWithSelectedFeatures;
}

//leave one out validator 
double leave_one_out_cross_validation(Classifier &classifier,const vector<Data> &data) {
  int correctPredictions = 0;

  for (size_t i = 0; i < data.size(); i++) {
    vector<Data> trainingData = data;
    Data testInstance = data[i];
    trainingData.erase(trainingData.begin() + i);

    classifier.Train(trainingData);
    int predictedClass = classifier.Test(testInstance);

    if (predictedClass == testInstance.classLabel) {
      correctPredictions++;
    }
  }

  return static_cast<double>(correctPredictions) / data.size();
}

//normalization through Z-score
void ZnormalizeData(vector<Data> &data) {
  size_t featureCount = data[0].features.size();

  for (size_t i = 0; i < featureCount; i++) {
    double sum = 0.0;
    for (auto &instance : data) {
      sum += instance.features[i];
    }

    double mean = sum / data.size();
    double sumOfSquaredDifferences = 0.0;

    for (auto &instance : data) {
      sumOfSquaredDifferences += pow((instance.features[i] - mean), 2);
    }

    double stdev = sqrt(sumOfSquaredDifferences / data.size());

    for (auto &instance : data) {
      instance.features[i] = (instance.features[i] - mean) / stdev;
    }
  }
}

//for features
string featuresToString(const vector<int> &features) {
  stringstream ss;
  ss << "{";
  for (size_t i = 0; i < features.size(); i++) {
    if (i != 0)
      ss << ",";
    ss << features[i];
  }
  ss << "}";
  return ss.str();
}

//accuracy calc
double calculateAccuracy(Classifier &classifier, vector<Data> &data,
                         const vector<int> &features) {
  vector<Data> dataset = SelectFeatures(data, features);
  return leave_one_out_cross_validation(classifier, dataset);
}

//default rate stuff, starting from part 3
double calculateDefaultRate(vector<Data> &data) {
    map<int, int> classCounts;

    for (const auto &instance : data) {
        classCounts[instance.classLabel]++;
    }

    int maxCount = 0;
    for (const auto &pair : classCounts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
        }
    }

    double defaultRate = static_cast<double>(maxCount) / data.size();
    return defaultRate;
}


// printing stuff 
string vectorToString(const vector<int> &vec) {
    string s = "{";
    for (int i = 0; i < vec.size(); ++i) {
        s += to_string(vec[i]);
        if (i != vec.size() - 1) s += ",";
    }
    s += "}";
    return s;
}

//forward search algo, choice 1
vector<int> ForwardSearch(Classifier &classifier, vector<Data> &data) {
    vector<int> currentSetOfFeatures; 
    vector<int> bestSoFarFeatureSet;
    double bestAccuracySoFar = 0; 
    //class attribute not included
    int totalNumberOfFeatures = data[0].features.size() - 1;
    bool improved = true;

    while(improved) {
        improved = false;
        int featureToAdd = -1;
        double bestSoFar = bestAccuracySoFar;
        
        for (int k = 1; k <= totalNumberOfFeatures; k++) {
            if (find(currentSetOfFeatures.begin(), currentSetOfFeatures.end(), k) == currentSetOfFeatures.end()) {
                vector<int> tempSetOfFeatures = currentSetOfFeatures;
                tempSetOfFeatures.push_back(k);
                double accuracy = calculateAccuracy(classifier, data, tempSetOfFeatures);
                
                cout << "Using feature(s) " << vectorToString(tempSetOfFeatures) << " accuracy is " << fixed << setprecision(1) << 
                accuracy * 100 << "%" << endl;
                if (accuracy > bestSoFar) {
                    bestSoFar = accuracy;
                    featureToAdd = k;
                }
            }
        }

        if (featureToAdd != -1) {
            currentSetOfFeatures.push_back(featureToAdd);
            bestAccuracySoFar = bestSoFar;
            bestSoFarFeatureSet = currentSetOfFeatures;
            improved = true;
            cout << "\n" << endl;
            cout << "Feature set " << vectorToString(currentSetOfFeatures) << " was best, accuracy is " << fixed << 
             setprecision(1) << bestSoFar * 100 << "%" << endl;
        }
    }

    cout << "\n" << endl;
    cout << "After finishing a complete forward search, the best feature subset is " 
     << vectorToString(bestSoFarFeatureSet) 
     << ", it has an accuracy of " << fixed << setprecision(1) << bestAccuracySoFar * 100 << "%" << endl;
    return bestSoFarFeatureSet;
}

//backward elim algo, choice 2
vector<int> BackwardElimination(Classifier &classifier, vector<Data> &data) {
    //class attribute not included
    int totalNumberOfFeatures = data[0].features.size() - 1;
    vector<int> currentSetOfFeatures(totalNumberOfFeatures); 
    vector<int> bestSoFarFeatureSet;
    double bestAccuracySoFar = 0; 

    
    for (int i = 1; i <= totalNumberOfFeatures; ++i) {
        currentSetOfFeatures[i - 1] = i;
    }

    for (int i = totalNumberOfFeatures; i > 0; i--) {
        int featureToRemove = -1;
        double bestSoFar = 0;
        
        for (int k = 0; k < currentSetOfFeatures.size(); k++) {
            vector<int> tempSetOfFeatures = currentSetOfFeatures;
            tempSetOfFeatures.erase(tempSetOfFeatures.begin() + k);
            double accuracy = calculateAccuracy(classifier, data, tempSetOfFeatures);

            cout << "Using feature(s) " << vectorToString(tempSetOfFeatures) << " accuracy is " << fixed << setprecision(1) << accuracy * 
             100 
              << "%" << endl;

            if (accuracy > bestSoFar) {
                bestSoFar = accuracy;
                featureToRemove = k;
            }
        }

        if (featureToRemove != -1) {
            currentSetOfFeatures.erase(currentSetOfFeatures.begin() + featureToRemove);
            cout << "\n" << endl;
            cout << "Feature set " << vectorToString(currentSetOfFeatures) << " was best, accuracy is " << 
              fixed << setprecision(1) << bestSoFar *100 << "%" << endl;
            if (bestSoFar > bestAccuracySoFar) {
                bestAccuracySoFar = bestSoFar;
                bestSoFarFeatureSet = currentSetOfFeatures;
            }
            else {
                cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
            }
        }
    }
    cout << "\n" << endl; 
    cout << "After finishing a complete backward search, the best feature subset is " << 
    vectorToString(bestSoFarFeatureSet) << " with an accuracy of " << fixed << setprecision(1) << bestAccuracySoFar * 100 << "%" << endl;
  return bestSoFarFeatureSet;
}

//Group 10's algo, is based off of the recursive elimation algorithm
vector<int> Group10Algo(Classifier &classifier, vector<Data> &data) {
    //class attribute not included
    int totalNumberOfFeatures = data[0].features.size() - 1;
    vector<int> currentSetOfFeatures;
   
    for (int i = 1; i <= totalNumberOfFeatures; ++i) {
        currentSetOfFeatures.push_back(i);
    }

    double bestAccuracySoFar = calculateAccuracy(classifier, data, currentSetOfFeatures);
    vector<int> bestSoFarFeatureSet = currentSetOfFeatures;

    while (currentSetOfFeatures.size() > 1) {
        int featureToRemove = -1;
        double smallestDrop = numeric_limits<double>::infinity();

        for (int j = 0; j < currentSetOfFeatures.size(); ++j) {
            vector<int> tempSetOfFeatures = currentSetOfFeatures;
            tempSetOfFeatures.erase(tempSetOfFeatures.begin() + j);
            double accuracy = calculateAccuracy(classifier, data, tempSetOfFeatures);
            double drop = bestAccuracySoFar - accuracy;

            cout << "Trying feature(s) " << vectorToString(tempSetOfFeatures) << " accuracy is " << fixed << setprecision(1) << 
              accuracy * 100 << "%" << endl;
            if (drop < smallestDrop) {
                smallestDrop = drop;
                featureToRemove = j;
            }
        }

        if (featureToRemove != -1) {
            currentSetOfFeatures.erase(currentSetOfFeatures.begin() + featureToRemove);
            double newAccuracy = calculateAccuracy(classifier, data, currentSetOfFeatures);
            if (newAccuracy > bestAccuracySoFar) {
                bestAccuracySoFar = newAccuracy;
                bestSoFarFeatureSet = currentSetOfFeatures;
            }
            cout << "\n" << endl;
            cout << "Removing feature " << featureToRemove << ", accuracy is now " << fixed << setprecision(1) << newAccuracy * 
             100 << "%" << endl;
        } else {
            break;
        }
    }

    cout << "\n" << endl;
    cout << "Through all the subsets, the best feature set is " << vectorToString(bestSoFarFeatureSet) << " with an accuracy of " << 
      fixed << setprecision(1) << bestAccuracySoFar * 100 << "%" << endl;
    return bestSoFarFeatureSet;
}

//main output interface 
int main() {
  Classifier classifier;
  vector<Data> data;
  string filename;
  vector<vector<double>> correlationMatrix;
  
  cout << "Welcome to Group #10 Feature Selection Algorithm." << endl;
  cout << "Type in the name of the file to test : ";
  cin >> filename;

  data = loadData(filename);
  cout << "\nThis dataset has " << data[0].features.size() - 1
       << " features (not including the class attribute), with " << data.size()
       << " instances." << endl;

  cout << "Please wait while I normalize the data... ";
  ZnormalizeData(data);
  cout << "Done!" << endl;

  // Calculate and print default rate
  double defaultRate = calculateDefaultRate(data);
  cout << "\n" << endl;
  cout << "Running nearest neighbor with no features (default rate), using leaving-one-out evaluation, get an accuracy of " << defaultRate * 100 << "%" << endl;


  cout << "\n" << endl;
  cout << "Type the number of the algorithm you want to run." << endl;
  cout << "1. Forward Selection" << endl;
  cout << "2. Backward Elimination" << endl;
  cout << "3. Group 10s Special Algorithm." << endl;
  int algoChoice;
  cin >> algoChoice;

  vector<int> bestFeatures;
  switch (algoChoice) {
  case 1:
    bestFeatures = ForwardSearch(classifier, data);
    break;
  case 2:
    bestFeatures = BackwardElimination(classifier, data);
    break;
  case 3:
    bestFeatures = Group10Algo(classifier, data);
    break;
  default:
    cout << "Invalid algorithm choice." << endl;
    return 1;
  }

  return 0;
}