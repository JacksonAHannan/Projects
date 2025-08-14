# Shiny dashboard for Online Retail II
# Drop this file into the project folder and run with shiny::runApp('app.R')

library(shiny)
library(tidyverse)
library(lubridate)
library(plotly)
library(DT)
library(scales)
library(shinyWidgets)
library(stringr)

# Path to CSV (update if moved)
data_path <- "/Users/jacksonhannan/Desktop/Python Projects/Sales Visualizations/online_retail_II copy.csv"

# Safe CSV read with guesswork for date formats
safe_read <- function(path){
  cat("Loading CSV file:", path, "\n")
  cat("This may take a moment for large files...\n")
  
  # Read CSV with faster settings for large files
  df <- readr::read_csv(path, 
                        guess_max = 10000,  # Reduced for faster loading
                        show_col_types = FALSE, 
                        locale = readr::locale(encoding = "UTF-8"),
                        skip_empty_rows = TRUE,
                        progress = TRUE)  # Show progress bar
  
  cat("Loaded", nrow(df), "rows and", ncol(df), "columns\n")
  
  # Quick header row cleanup - only check last few rows for efficiency
  if(nrow(df) > 1000) {
    # Check last 100 rows for header contamination
    check_rows <- max(1, nrow(df) - 99):nrow(df)
    header_keywords <- c("invoice", "quantity", "price", "customer", "description", "country", "stock", "date")
    
    header_rows <- c()
    for(i in check_rows) {
      if(i <= nrow(df)) {
        row_text <- tolower(paste(as.character(df[i, ]), collapse = " "))
        if(any(sapply(header_keywords, function(x) grepl(paste0("\\b", x), row_text)))) {
          header_rows <- c(header_rows, i)
        }
      }
    }
    
    if(length(header_rows) > 0) {
      cat("Removing", length(header_rows), "header-like rows from end of file\n")
      df <- df[-header_rows, ]
    }
  }
  
  cat("Data cleaning complete. Final dataset:", nrow(df), "rows\n")
  
  # Normalize column names
  names(df) <- make.names(names(df))
  # Common column name guesses
  colnames(df) <- gsub("^InvoiceNo$|^Invoice$","InvoiceNo", colnames(df), ignore.case = TRUE)
  colnames(df) <- gsub("^InvoiceDate$|^Date$","InvoiceDate", colnames(df), ignore.case = TRUE)
  colnames(df) <- gsub("^Description$|^ItemDescription$","Description", colnames(df), ignore.case = TRUE)
  colnames(df) <- gsub("^Quantity$","Quantity", colnames(df), ignore.case = TRUE)
  colnames(df) <- gsub("^UnitPrice$|^Price$","UnitPrice", colnames(df), ignore.case = TRUE)
  colnames(df) <- gsub("^Country$","Country", colnames(df), ignore.case = TRUE)

  # If InvoiceDate exists, try parsing
  if("InvoiceDate" %in% colnames(df)){
    cat("Parsing dates...\n")
    cat("Sample date values:", head(df$InvoiceDate, 10), "\n")
    
    # Simplified robust date parsing for speed
    robust_parse_date <- function(x){
      x2 <- trimws(as.character(x))
      
      # Try the format we actually see: MM/D/YY H:MM
      parsed <- suppressWarnings(
        parse_date_time(x2, orders = c('mdy HM', 'mdy HMS', 'md y HM', 'md y HMS'), tz = 'UTC', quiet = TRUE)
      )
      
      # If that didn't work, try other common formats
      if(all(is.na(parsed))) {
        parsed <- suppressWarnings(
          parse_date_time(x2, orders = c('ymd HMS', 'dmy HMS', 'ymd HM', 'dmy HM'), tz = 'UTC', quiet = TRUE)
        )
      }
      
      # Simple fallback for remaining NAs
      nas <- is.na(parsed)
      if(any(nas) && sum(nas) < 1000) {  # Only do expensive parsing for small number of NAs
        # Try Excel dates
        num <- suppressWarnings(as.numeric(x2[nas]))
        excel_idx <- which(!is.na(num) & num > 25000 & num < 100000)
        if(length(excel_idx)){
          parsed[nas][excel_idx] <- as.POSIXct(as.Date(num[excel_idx], origin = '1899-12-30'), tz = 'UTC')
        }
      }
      
      parsed
    }
    
    df <- df %>% mutate(
      InvoiceDate_parsed = robust_parse_date(InvoiceDate)
    )
    
    # Check parsing success
    successful_dates <- sum(!is.na(df$InvoiceDate_parsed))
    total_dates <- nrow(df)
    cat("Date parsing complete:", successful_dates, "of", total_dates, "dates parsed successfully\n")
    
    if(successful_dates > 0) {
      df <- df %>% rename(InvoiceDate_raw = InvoiceDate, InvoiceDate = InvoiceDate_parsed)
    } else {
      cat("WARNING: No dates could be parsed. Creating dummy date column.\n")
      df$InvoiceDate <- as.POSIXct("2023-01-01", tz = 'UTC')
    }
  }

  # Ensure numeric columns
  if(!"Quantity" %in% colnames(df)) df$Quantity <- 0
  if(!"UnitPrice" %in% colnames(df)) df$UnitPrice <- 0
  df <- df %>% mutate(
    Quantity = suppressWarnings(as.numeric(Quantity)),
    UnitPrice = suppressWarnings(as.numeric(UnitPrice)),
    Sales = coalesce(Quantity,0) * coalesce(UnitPrice,0),
    Description = if("Description" %in% colnames(df)) as.character(Description) else NA_character_,
    Country = if("Country" %in% colnames(df)) as.character(Country) else NA_character_,
    InvoiceNo = if("InvoiceNo" %in% colnames(df)) as.character(InvoiceNo) else NA_character_
  )

  # Type: mark returns if invoice starts with 'C' (common in Online Retail dataset)
  df <- df %>% mutate(Type = ifelse(grepl('^C', InvoiceNo, ignore.case = TRUE),'Return','Sale'))
  
  # Calculate Net Sales: positive for sales, negative for returns
  df <- df %>% mutate(
    NetSales = case_when(
      Type == 'Sale' ~ Sales,
      Type == 'Return' ~ -abs(Sales),  # Make returns negative
      TRUE ~ Sales
    )
  )

  # YearMonth for grouping
  if("InvoiceDate" %in% colnames(df)){
    df <- df %>% mutate(YearMonth = floor_date(InvoiceDate, unit = 'month'))
  } else {
    df$YearMonth <- as.Date(NA)
  }

  # Comprehensive country analysis
  if("Country" %in% colnames(df)) {
    country_summary <- df %>% 
      group_by(Country) %>% 
      summarise(count = n(), .groups = 'drop') %>%
      arrange(desc(count))
    
    cat("Country analysis:\n")
    cat("Total unique countries (including NA):", nrow(country_summary), "\n")
    cat("Top 20 countries by record count:\n")
    print(head(country_summary, 20))
    
    # Non-NA countries
    non_na_countries <- country_summary %>% filter(!is.na(Country) & Country != "")
    cat("Non-NA countries:", nrow(non_na_countries), "\n")
    if(nrow(non_na_countries) > 0) {
      cat("Non-NA country list:", paste(head(non_na_countries$Country, 15), collapse = ", "), "\n")
    }
  }

  df
}

raw <- safe_read(data_path)

# Basic checks
if(nrow(raw) == 0){
  stop('Dataframe is empty after reading. Check the CSV path and file contents: ', data_path)
}

cat("Dashboard ready! Dataset loaded with", nrow(raw), "rows and", ncol(raw), "columns\n")
cat("Starting Shiny app...\n")

ui <- fluidPage(
  titlePanel('Online Retail II — Sales Dashboard'),
  sidebarLayout(
    sidebarPanel(
      width = 3,
      helpText('Filters to slice the data'),
      uiOutput('date_ui'),
      pickerInput('countries', 'Country (multi-select)', choices = NULL, options = list(`actions-box` = TRUE), multiple = TRUE),
      pickerInput('types', 'Type', choices = c('All','Sale','Return'), selected = c('All'), multiple = FALSE),
      textInput('item_search', 'Search item description (substring)', ''),
      sliderInput('top_n', 'Top N items to show', min = 5, max = 50, value = 15),
      br(),
      radioButtons('sales_metric', 'Sales Metric:', 
                   choices = list('Total Sales' = 'total', 'Net Sales (Sales - Returns)' = 'net'), 
                   selected = 'net'),
      br(),
      actionButton('update_viz', 'Update Visualizations', class = 'btn-primary', style = 'width: 100%;'),
      br(), br(),
      downloadButton('download_filtered','Download filtered CSV', style = 'width: 100%;')
    ),
    mainPanel(
      tabsetPanel(
        tabPanel('Overview',
                 fluidRow(
                   column(3, wellPanel(h4('Total Sales'), h3(textOutput('total_sales')))),
                   column(3, wellPanel(h4('Net Sales'), h3(textOutput('net_sales')))),
                   column(3, wellPanel(h4('Total Orders'), h3(textOutput('total_orders')))),
                   column(3, wellPanel(h4('Avg Order Value'), h3(textOutput('aov'))))
                 ),
                 hr(),
                 plotlyOutput('monthly_ts', height = '350px')
        ),
        tabPanel('By Country',
                 fluidRow(
                   column(6, plotlyOutput('country_bar', height = '500px')),
                   column(6, plotlyOutput('country_pie', height = '500px'))
                 )
        ),
        tabPanel('By Item',
                 plotlyOutput('item_bar', height = '500px'),
                 hr(),
                 DTOutput('item_table')
        ),
        tabPanel('Raw Data',
                 DTOutput('raw_table')
        )
      )
    )
  )
)

server <- function(input, output, session){
  # initialize filter choices only once
  observeEvent(raw, {
    if('Country' %in% colnames(raw)){
      # Get all non-NA, non-empty countries
      all_countries <- raw %>% 
        filter(!is.na(Country) & Country != "" & Country != "NA") %>%
        pull(Country) %>%
        unique() %>%
        sort()
      
      cat("Setting up country filter with", length(all_countries), "countries\n")
      if(length(all_countries) > 0) {
        cat("Countries for filter:", paste(head(all_countries, 10), collapse = ", "), 
            if(length(all_countries) > 10) "..." else "", "\n")
        # Select top 5 countries by default, or all if fewer than 10
        default_selection <- if(length(all_countries) <= 10) all_countries else head(all_countries, 5)
        updatePickerInput(session, 'countries', choices = all_countries, selected = default_selection)
      } else {
        updatePickerInput(session, 'countries', choices = c("No countries available"), selected = NULL)
      }
    }
  }, once = TRUE)

  output$date_ui <- renderUI({
    if('InvoiceDate' %in% colnames(raw) && any(!is.na(raw$InvoiceDate))){
      min_d <- min(raw$InvoiceDate, na.rm = TRUE)
      max_d <- max(raw$InvoiceDate, na.rm = TRUE)
      
      # Check if we got valid dates
      if(is.finite(min_d) && is.finite(max_d)) {
        dateRangeInput('date_range', 'Date range', start = min_d, end = max_d)
      } else {
        tags$div(
          tags$p('Date parsing issues detected'),
          tags$p('Using all available data')
        )
      }
    } else {
      tags$p('No valid InvoiceDate found in dataset')
    }
  })

  filtered <- eventReactive(input$update_viz, {
    df <- raw
    # date filter
    if(!is.null(input$date_range) && 'InvoiceDate' %in% colnames(df)){
      start <- as.POSIXct(input$date_range[1], tz = 'UTC')
      end <- as.POSIXct(input$date_range[2] + days(1) - seconds(1), tz = 'UTC')
      df <- df %>% filter(InvoiceDate >= start & InvoiceDate <= end)
    }
    # country filter
    if(!is.null(input$countries) && length(input$countries) > 0){
      df <- df %>% filter(Country %in% input$countries)
    }
    # type filter
    if(!is.null(input$types) && input$types != 'All'){
      df <- df %>% filter(Type == input$types)
    }
    # item search
    if(!is.null(input$item_search) && input$item_search != ''){
      df <- df %>% filter(str_detect(tolower(coalesce(Description,'')), tolower(input$item_search)))
    }
    df
  }, ignoreNULL = FALSE)
  
  # Create a reactive that provides initial data when button hasn't been clicked
  filtered_data <- reactive({
    if(input$update_viz == 0) {
      # Return all data initially
      return(raw)
    } else {
      return(filtered())
    }
  })

  # KPIs
  output$total_sales <- renderText({
    f <- filtered_data()
    dollar(sum(f$Sales, na.rm = TRUE))
  })
  output$net_sales <- renderText({
    f <- filtered_data()
    # Calculate net sales: Sales from 'Sale' type minus absolute value of 'Return' sales
    sales_amount <- f %>% filter(Type == 'Sale') %>% summarise(total = sum(Sales, na.rm = TRUE)) %>% pull(total)
    returns_amount <- f %>% filter(Type == 'Return') %>% summarise(total = sum(abs(Sales), na.rm = TRUE)) %>% pull(total)
    net_sales <- ifelse(length(sales_amount) == 0, 0, sales_amount) - ifelse(length(returns_amount) == 0, 0, returns_amount)
    dollar(net_sales)
  })
  output$total_orders <- renderText({
    f <- filtered_data()
    # Count distinct invoices for proper order count
    total_orders <- f %>% 
      filter(!is.na(InvoiceNo) & InvoiceNo != "") %>%
      summarise(orders = n_distinct(InvoiceNo)) %>%
      pull(orders)
    formatC(total_orders, format = 'd', big.mark = ',')
  })
  output$unique_customers <- renderText({
    f <- filtered_data()
    if('CustomerID' %in% colnames(f)){
      formatC(n_distinct(f$CustomerID), format = 'd', big.mark = ',')
    } else {
      'N/A'
    }
  })
  output$aov <- renderText({
    f <- filtered_data()
    orders <- f %>% group_by(InvoiceNo) %>% summarise(order_value = sum(Sales, na.rm = TRUE))
    if(nrow(orders) == 0) return('$0')
    dollar(mean(orders$order_value, na.rm = TRUE))
  })

  # Monthly time series
  output$monthly_ts <- renderPlotly({
    f <- filtered_data()
    if(!'YearMonth' %in% colnames(f) || nrow(f) == 0) {
      # Return empty plot with message
      p <- ggplot(data.frame(x = 1, y = 1, label = "No date data available")) +
        geom_text(aes(x, y, label = label)) +
        theme_void() +
        labs(title = "Date parsing issues - unable to create time series")
      return(ggplotly(p))
    }
    
    # Check if we have valid dates
    valid_dates <- !is.na(f$YearMonth)
    if(!any(valid_dates)) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No valid dates found")) +
        geom_text(aes(x, y, label = label)) +
        theme_void() +
        labs(title = "All dates are NA - check date format in CSV")
      return(ggplotly(p))
    }
    
    # Choose sales metric based on user selection
    sales_col <- if(input$sales_metric == 'net') 'NetSales' else 'Sales'
    metric_label <- if(input$sales_metric == 'net') 'Net Sales' else 'Total Sales'
    
    monthly <- f %>% 
      filter(!is.na(YearMonth)) %>%
      group_by(YearMonth) %>% 
      summarise(SalesValue = sum(.data[[sales_col]], na.rm = TRUE), .groups = 'drop') %>% 
      arrange(YearMonth)
    
    if(nrow(monthly) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No data to display")) +
        geom_text(aes(x, y, label = label)) +
        theme_void()
      return(ggplotly(p))
    }
    
    p <- ggplot(monthly, aes(x = YearMonth, y = SalesValue)) +
      geom_line(color = '#2c7fb8') + geom_point(color = '#2c7fb8') +
      scale_y_continuous(labels = dollar) +
      labs(x = '', y = metric_label, title = paste('Monthly', tolower(metric_label), '(filtered)')) + theme_minimal()
    ggplotly(p, tooltip = c('x','y'))
  })

  # Country bar
  output$country_bar <- renderPlotly({
    f <- filtered_data()
    if(nrow(f) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No data available")) +
        geom_text(aes(x, y, label = label)) +
        theme_void()
      return(ggplotly(p))
    }
    
    # Choose sales metric based on user selection
    sales_col <- if(input$sales_metric == 'net') 'NetSales' else 'Sales'
    metric_label <- if(input$sales_metric == 'net') 'Net Sales' else 'Total Sales'
    
    # Analyze all countries including handling missing data
    country_s <- f %>% 
      mutate(
        Country_clean = case_when(
          is.na(Country) | Country == "" ~ "Unknown/Missing",
          TRUE ~ as.character(Country)
        )
      ) %>%
      group_by(Country_clean) %>% 
      summarise(
        SalesValue = sum(.data[[sales_col]], na.rm = TRUE), 
        Orders = n_distinct(InvoiceNo),
        Records = n(),
        .groups = 'drop'
      ) %>% 
      arrange(desc(SalesValue)) %>%
      slice_head(n = 25)  # Show top 25 countries
    
    if(nrow(country_s) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No data to display")) +
        geom_text(aes(x, y, label = label)) +
        theme_void() +
        labs(title = "No country data available")
      return(ggplotly(p))
    }
    
    # Create interactive bar chart
    p <- ggplot(country_s, aes(x = reorder(Country_clean, SalesValue), y = SalesValue, 
                               text = paste0("Country: ", Country_clean, "<br>",
                                           metric_label, ": ", scales::dollar(SalesValue), "<br>",
                                           "Orders: ", scales::comma(Orders), "<br>",
                                           "Records: ", scales::comma(Records)))) +
      geom_col(fill = '#7fc97f') + 
      coord_flip() + 
      scale_y_continuous(labels = scales::dollar_format()) +
      labs(x = '', y = metric_label, title = paste('Top', nrow(country_s), 'Countries by', metric_label)) + 
      theme_minimal() +
      theme(axis.text.y = element_text(size = 10))
    
    ggplotly(p, tooltip = "text")
  })

  # Country pie chart
  output$country_pie <- renderPlotly({
    f <- filtered_data()
    if(nrow(f) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No data available")) +
        geom_text(aes(x, y, label = label)) +
        theme_void()
      return(ggplotly(p))
    }
    
    # Choose sales metric based on user selection
    sales_col <- if(input$sales_metric == 'net') 'NetSales' else 'Sales'
    metric_label <- if(input$sales_metric == 'net') 'Net Sales' else 'Total Sales'
    
    # Get top 10 countries for pie chart (pie charts get cluttered with too many slices)
    country_s <- f %>% 
      mutate(
        Country_clean = case_when(
          is.na(Country) | Country == "" ~ "Unknown/Missing",
          TRUE ~ as.character(Country)
        )
      ) %>%
      group_by(Country_clean) %>% 
      summarise(
        SalesValue = sum(.data[[sales_col]], na.rm = TRUE), 
        Orders = n_distinct(InvoiceNo),
        Records = n(),
        .groups = 'drop'
      ) %>% 
      arrange(desc(SalesValue)) %>%
      slice_head(n = 10)  # Top 10 for pie chart
    
    if(nrow(country_s) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No data to display")) +
        geom_text(aes(x, y, label = label)) +
        theme_void() +
        labs(title = "No country data available")
      return(ggplotly(p))
    }
    
    # Create pie chart
    p <- plot_ly(country_s, 
                 labels = ~Country_clean, 
                 values = ~SalesValue, 
                 type = 'pie',
                 textinfo = 'label+percent',
                 textposition = 'outside',
                 hovertemplate = paste0(
                   "<b>%{label}</b><br>",
                   metric_label, ": %{value:$,.0f}<br>",
                   "Percentage: %{percent}<br>",
                   "<extra></extra>"
                 )) %>%
      layout(title = paste('Top', nrow(country_s), 'Countries by', metric_label, '(Pie Chart)'),
             showlegend = TRUE,
             legend = list(orientation = "v", x = 1.02, y = 0.5))
    
    return(p)
  })

  # Item metrics
  output$item_bar <- renderPlotly({
    f <- filtered_data()
    if(nrow(f) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No data available")) +
        geom_text(aes(x, y, label = label)) +
        theme_void()
      return(ggplotly(p))
    }
    
    # Choose sales metric based on user selection
    sales_col <- if(input$sales_metric == 'net') 'NetSales' else 'Sales'
    metric_label <- if(input$sales_metric == 'net') 'Net Sales' else 'Total Sales'
    
    items <- f %>% 
      filter(!is.na(Description) & Description != "") %>%
      group_by(Description) %>% 
      summarise(
        SalesValue = sum(.data[[sales_col]], na.rm = TRUE), 
        Quantity = sum(Quantity, na.rm = TRUE), 
        .groups = 'drop'
      ) %>% 
      arrange(desc(SalesValue)) %>% 
      slice_head(n = input$top_n)
    
    if(nrow(items) == 0) {
      p <- ggplot(data.frame(x = 1, y = 1, label = "No item data available")) +
        geom_text(aes(x, y, label = label)) +
        theme_void()
      return(ggplotly(p))
    }
    
    p <- ggplot(items, aes(x = reorder(Description, SalesValue), y = SalesValue, text = paste('Qty:', Quantity))) +
      geom_col(fill = '#beaed4') + coord_flip() + scale_y_continuous(labels = dollar) +
      labs(x = '', y = metric_label, title = paste('Top', input$top_n, 'items by', tolower(metric_label))) + theme_minimal()
    ggplotly(p, tooltip = c('y','text','x'))
  })

  output$item_table <- renderDT({
    f <- filtered_data()
    # Choose sales metric based on user selection
    sales_col <- if(input$sales_metric == 'net') 'NetSales' else 'Sales'
    metric_label <- if(input$sales_metric == 'net') 'Net Sales' else 'Total Sales'
    
    items <- f %>% group_by(StockCode = ifelse('StockCode' %in% colnames(f), StockCode, NA), Description) %>%
      summarise(
        !!metric_label := sum(.data[[sales_col]], na.rm = TRUE), 
        Quantity = sum(Quantity, na.rm = TRUE), 
        Orders = n_distinct(InvoiceNo), 
        .groups = 'drop'
      ) %>%
      arrange(desc(.data[[metric_label]]))
    datatable(items, options = list(pageLength = 10, scrollX = TRUE))
  })

  output$raw_table <- renderDT({
    f <- filtered_data()
    # show a subset of columns to keep table readable
    show_cols <- intersect(c('InvoiceNo','InvoiceDate','Description','StockCode','Quantity','UnitPrice','Sales','NetSales','Country','CustomerID','Type','YearMonth'), colnames(f))
    datatable(f %>% select(all_of(show_cols)) %>% mutate_if(is.POSIXt, ~as.character(.)), options = list(pageLength = 25, scrollX = TRUE))
  })

  output$download_filtered <- downloadHandler(
    filename = function(){ paste0('filtered_data_', Sys.Date(), '.csv') },
    content = function(file){
      readr::write_csv(filtered_data(), file)
    }
  )
}

shinyApp(ui, server)
