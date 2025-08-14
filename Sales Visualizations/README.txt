# UK Retailer Sales Dashboard

## Overview
This is an interactive Shiny dashboard for analyzing online retail sales data. The dashboard provides comprehensive visualizations and analytics for exploring sales performance across different dimensions including time, geography, and product categories.

## Data Source
The data used in this dashboard comes from the UCI Machine Learning Repository:
**Online Retail II Dataset**
URL: https://archive.ics.uci.edu/dataset/502/online+retail+ii

This dataset contains transactional data for a UK-based online retailer, covering the period from 2009-2011. It includes information about invoices, products, quantities, prices, customers, and countries.

## Features

### Dashboard Tabs:
1. **Overview Tab**
   - Key Performance Indicators (KPIs): Total Sales, Net Sales, Total Orders, Average Order Value
   - Monthly time series chart showing sales trends over time

2. **By Country Tab**
   - Interactive bar chart showing top countries by sales/net sales
   - Pie chart displaying country sales distribution
   - Support for 44+ countries in the dataset

3. **By Item Tab**
   - Top N items bar chart (configurable from 5-50 items)
   - Detailed item table with sales, quantity, and order information

4. **Raw Data Tab**
   - Complete dataset view with filtering capabilities
   - Export functionality for filtered data

### Key Features:
- **Sales Metric Toggle**: Switch between Total Sales and Net Sales (Sales minus Returns)
- **Advanced Filtering**: Date range, country, transaction type, and item search
- **Interactive Visualizations**: Hover tooltips, zoom, and pan capabilities
- **Real-time Updates**: All charts update dynamically based on filters
- **Data Export**: Download filtered datasets as CSV files

### Returns Analysis:
- Automatic detection of return transactions (invoices starting with 'C')
- Net Sales calculation: Sales - |Returns|
- Return impact analysis across all visualizations

## Technical Requirements

### R Packages Required:
- shiny
- tidyverse
- lubridate
- plotly
- DT
- scales
- shinyWidgets
- stringr

### Installation:
```r
install.packages(c("shiny", "tidyverse", "lubridate", "plotly", 
                   "DT", "scales", "shinyWidgets", "stringr"))
```

## Setup Instructions

1. **Download Data**: 
   - Download the Online Retail II dataset from UCI repository
   - Save as CSV file in the project directory
   - Update the `data_path` variable in the R script to point to your CSV file

2. **File Structure**:
   ```
   Project Directory/
   ├── UK_Retailer_Visualization.R (or app.R)
   ├── online_retail_II copy.csv (your data file)
   └── README.txt (this file)
   ```

3. **Run the Dashboard**:
   ```r
   # Option 1: If file is named app.R
   shiny::runApp()
   
   # Option 2: If file has different name
   shiny::runApp('UK_Retailer_Visualization.R')
   ```

4. **Access**: Open your web browser and navigate to the displayed URL (typically http://127.0.0.1:xxxx)

## Data Processing Features

### Automatic Data Cleaning:
- Date parsing with multiple format support
- Column name normalization
- Header row detection and removal
- Missing value handling

### Date Parsing:
- Supports multiple date formats (MM/D/YY H:MM, Excel dates, etc.)
- Robust fallback mechanisms for date parsing failures
- Time zone standardization (UTC)

### Performance Optimizations:
- Large file handling with progress indicators
- Efficient memory usage for 1M+ records
- Optimized filtering and aggregation

## Dataset Statistics
- **Total Records**: ~1,067,372 transactions
- **Date Range**: 2009-2011
- **Countries**: 44 unique countries/regions
- **Primary Market**: United Kingdom (91.8% of transactions)
- **Transaction Types**: Sales and Returns
- **Products**: Thousands of unique items

## Usage Tips

1. **Performance**: For large datasets, use the "Update Visualizations" button after setting filters
2. **Country Analysis**: The dataset is heavily UK-focused, but includes significant data from EIRE, Germany, France, and other European countries
3. **Returns Analysis**: Use the Net Sales metric to understand true profitability after returns
4. **Time Analysis**: Filter by date ranges to analyze seasonal trends
5. **Product Analysis**: Use the item search to find specific product categories

## File Information
- **Main Script**: UK_Retailer_Visualization.R
- **Framework**: R Shiny
- **Visualization Library**: Plotly (interactive charts)
- **Data Manipulation**: tidyverse/dplyr
- **Date Handling**: lubridate

## License
This code is provided for educational and analytical purposes. Please refer to the UCI dataset license for data usage terms.

## Contact
For questions or issues with this dashboard, please refer to the code comments or R community resources.

---
Last Updated: August 2025
Dashboard Version: 2.0 (with Net Sales functionality)
