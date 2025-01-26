# Simple Moving Average (SMA) Time Series Model

## Overview

This project implements a Simple Moving Average (SMA) time series forecasting model to predict future values based on historical donation data. The model is designed to smooth out short-term fluctuations and highlight longer-term trends in the data.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Requirements

The following Python packages are required to run this code:

- `pandas`
- `numpy`
- `matplotlib`

You can install these packages using pip:


## Installation

1. Clone this repository to your local machine:


2. Install the required packages:

requirements.txt file using pip

## Usage

1. Prepare your dataset in CSV format with the following structure:

| Members Name          | 2024-01 | 2024-02 | 2024-03 | ... | 2024-11 |
|-----------------------|---------|---------|---------|-----|---------|
| Noor Fatima Sadaqat   | 0       | 500     | 600     | ... | 1390    |
| Zunaira Shafiq       | 2000    | 500     | 750     | ... | 9465    |
| Muneeb Salman         | 6946    | 0       | 2000    | ... | 71311   |
| ...                   | ...     | ...     | ...     | ... | ...     |

2. Update the path in the code where the dataset is loaded:

df = pd.read_csv('path_to_your_dataset.csv')


3. Run the script to generate predictions for the next month's donations:

python sma_forecast.py


4. The predicted values will be displayed, and plots will be generated for each member's donation trend.

## Dataset

The dataset should be structured as shown above. Ensure there are no NaN or non-numeric values in the dataset.

## Code Explanation

The main components of the code include:

1. **Data Preparation**: The dataset is loaded, transposed, and any missing values are filled with zeros.

2. **Simple Moving Average Calculation**: The SMA is calculated using a specified window size (default is 3 months). The last calculated moving average is used as the forecast for the next month.

3. **Visualization**: The actual donations and predicted values are plotted for each member, allowing for easy comparison of trends.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
