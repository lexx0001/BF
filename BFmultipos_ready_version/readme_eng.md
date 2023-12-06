# Application Structure

The `config.py` file accepts all input variables.

### BreakoutFinder
"Bull/Bear Cup" indicator - generates long and short signals.

### CSVHandler
Handles CSV file processing, extracts data into pandas, and performs time conversion.

### DCAorderCalc
Calculates values for DCA orders and provides data for analysis.

### CalcMultiposPrices
Calculates prices for multi-position modules.

### DCAandTPplaceCalc
Determines the placement of DCA orders and TP.

### MultiposRunnings
Tracks the operation and launch of multi-position modules.

# README.md

## Description
The application is designed for analyzing DCA (Dollar Cost Averaging) positions using the method of insurance orders.

In its current form, the application takes dataframes (candlestick charts) in CSV format, processes them using the built-in "Breakout Finger" indicator to generate short/long signals, and performs the necessary calculations for placing DCA positions. The distinctive feature is the strategy of multiple placement of DCA modules, which allows for more flexible strategy customization.

A DCA module is an unclosed DCA position, meaning one module can be considered one position, although it is a set of insurance orders.

## Installation
Python Version: 3.10.11
Libraries:
- pandas Version: 2.1.3
- pytz Version: 2023.3.post1
- os

## Usage
[Examples of using your classes, how to pass variables through `config.py`, and how to run various modules.]

## Example
[Example code or script demonstrating the basic usage of your application.]

## License
[Specify the license applicable to your code.]

## Author
Alexey Novikov
Telegram: @lexxloud
