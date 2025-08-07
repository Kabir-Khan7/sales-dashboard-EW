import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from PIL import Image
import base64
import chardet
import os

# Configure page
st.set_page_config(
    page_title="Electronic World - Professional Sales Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Try to load logo image
try:
    logo = Image.open('logo.png')  # Replace with your image file name
    logo_base64 = image_to_base64(logo)
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="120">'
except:
    logo_html = '📊'

# Custom CSS for professional styling
st.markdown(f"""
    <style>
    .main-title {{
        font-size: 2.3rem; 
        color: #2e86ab; 
        text-align: center;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }}
    .section-header {{
        font-size: 1.6rem; 
        color: #2e86ab; 
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{ 
        background-color: #f8f9fa; 
        border-radius: 8px; 
        padding: 1.2rem; 
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .dataframe {{
        font-family: Arial, sans-serif;
    }}
    .dataframe thead th {{
        background-color: #2e86ab;
        color: white;
        font-weight: bold;
        text-align: center;
    }}
    .dataframe tbody tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    .dataframe tbody tr:hover {{
        background-color: #e9ecef;
    }}
    .total-row {{
        background-color: #2e86ab !important;
        color: white !important;
        font-weight: bold;
    }}
    .positive {{
        color: #28a745;
    }}
    .negative {{
        color: #dc3545;
    }}
    .chart-container {{
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .arrow-up {{
        color: #28a745;
        font-weight: bold;
    }}
    .arrow-down {{
        color: #dc3545;
        font-weight: bold;
    }}
    .priority-high {{
        background-color: #ffcccc !important;
        font-weight: bold;
    }}
    .priority-medium {{
        background-color: #fff3cd !important;
        font-weight: bold;
    }}
    .priority-low {{
        background-color: #d4edda !important;
        font-weight: bold;
    }}
    /* Green color for filter widgets */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: #28a745 !important;
        color: white !important;
    }}
    .stSelectbox [data-baseweb="select"] {{
        border-color: #28a745 !important;
    }}
    .stCheckbox [data-baseweb="checkbox"] {{
        border-color: #28a745 !important;
    }}
    .stButton button {{
        background-color: #28a745 !important;
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Title with logo
st.markdown(f'<div class="main-title">{logo_html} Electronic World - Professional Sales Dashboard</div>', unsafe_allow_html=True)

def detect_month_columns(df):
    """Detect month columns in YY-Mon format (24-Oct, 25-Jan, etc.)"""
    month_cols = [col for col in df.columns if any(m in col for m in ['-Jan', '-Feb', '-Mar', '-Apr', '-May', '-Jun', 
                                                                    '-Jul', '-Aug', '-Sep', '-Oct', '-Nov', '-Dec'])]
    
    # Ensure we have all months from Oct-24 to Jul-25 in correct order
    expected_months = ['24-Jan', '24-Oct', '24-Nov', '24-Dec', '25-Jan', '25-Feb', 
                      '25-Mar', '25-Apr', '25-May', '25-Jun', '25-Jul']
    
    # Filter to only include expected months that exist in the dataframe
    month_cols = [m for m in expected_months if m in df.columns]
    
    return month_cols

def calculate_all_metrics(df, month_cols, selected_months=None):
    """Calculate all required metrics automatically"""
    # If specific months are selected, use only those
    if selected_months:
        month_cols = [m for m in month_cols if m in selected_months]
    
    # Ensure numeric columns
    numeric_cols = month_cols + [col for col in ['Store Landed Cost', 'Retail Price', 'Total Unit Ordered', 
                                               'Sold %', 'Quantity Refunded'] 
                               if col in df.columns]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Rename columns to match expected names
    if 'Store Landed Cost' in df.columns:
        df['Store Landed'] = df['Store Landed Cost']
    if 'Total Unit Sold' in df.columns:
        df['Total Units Sold'] = df['Total Unit Sold']
    if 'Sold %' in df.columns:
        df['Total Unit Sold %'] = df['Sold %']
    if 'Quantity Refunded' in df.columns:
        df['Quantity Refund'] = df['Quantity Refunded']
    
    # Calculate core metrics
    df['Total Units Sold'] = df[month_cols].sum(axis=1)
    if 'Retail Price' in df.columns:
        df['Total Revenue'] = (df[month_cols].sum(axis=1) * df['Retail Price'])
    if 'Store Landed' in df.columns:
        df['Total COGS'] = df[month_cols].sum(axis=1) * df['Store Landed']
    if 'Total Revenue' in df.columns and 'Total COGS' in df.columns:
        df['Gross Profit'] = df['Total Revenue'] - df['Total COGS']
        df['Gross Margin %'] = np.where(
            df['Total Revenue'] > 0,
            (df['Gross Profit'] / df['Total Revenue']) * 100,
            0
        )
    
    # Calculate refund metrics
    if 'Quantity Refund' in df.columns and 'Total Units Sold' in df.columns:
        df['Refund Rate %'] = np.where(
            df['Total Units Sold'] > 0,
            (df['Quantity Refund'] / df['Total Units Sold']) * 100,
            0
        )
    
    # Calculate monthly metrics if we have at least 2 months
    sorted_months = []
    if len(month_cols) >= 1:
        try:
            # Convert month columns to standard format for sorting (Oct-24)
            month_mapping = {m: f"{m.split('-')[1]}-{m.split('-')[0]}" for m in month_cols}
            sorted_months = sorted(month_cols, 
                                 key=lambda x: pd.to_datetime(month_mapping[x], format='%b-%y'))
            
            if len(sorted_months) >= 2:
                # Calculate MoM growth for each metric
                for i in range(1, len(sorted_months)):
                    current_month = sorted_months[i]
                    prev_month = sorted_months[i-1]
                    
                    # Units growth
                    df[f'{current_month} Units Growth %'] = np.where(
                        df[prev_month] > 0,
                        ((df[current_month] - df[prev_month]) / df[prev_month]) * 100,
                        np.where(df[current_month] > 0, 100, 0)
                    )
                    
                    # Revenue growth
                    if 'Retail Price' in df.columns:
                        df[f'{current_month} Revenue Growth %'] = np.where(
                            (df[prev_month] * df['Retail Price']) > 0,
                            ((df[current_month] * df['Retail Price'] - df[prev_month] * df['Retail Price']) / 
                             (df[prev_month] * df['Retail Price'])) * 100,
                            0
                        )
                    
                    # COGS growth
                    if 'Store Landed' in df.columns:
                        df[f'{current_month} COGS Growth %'] = np.where(
                            (df[prev_month] * df['Store Landed']) > 0,
                            ((df[current_month] * df['Store Landed'] - df[prev_month] * df['Store Landed']) / 
                             (df[prev_month] * df['Store Landed'])) * 100,
                            0
                        )
                    
                    # Gross Profit growth
                    if 'Retail Price' in df.columns and 'Store Landed' in df.columns:
                        current_gp = df[current_month] * (df['Retail Price'] - df['Store Landed'])
                        prev_gp = df[prev_month] * (df['Retail Price'] - df['Store Landed'])
                        df[f'{current_month} GP Growth %'] = np.where(
                            prev_gp > 0,
                            ((current_gp - prev_gp) / prev_gp) * 100,
                            0
                        )
            
            # Recent metrics (last 3 months)
            if len(sorted_months) >= 1:
                recent_months = sorted_months[-3:] if len(sorted_months) >= 3 else sorted_months
                df['Recent Sales'] = df[recent_months].sum(axis=1)
                df['Sales Velocity (units/day)'] = df['Recent Sales'] / (90 if len(recent_months) == 3 else 30*len(recent_months))
                
                if len(recent_months) >= 2:
                    df['3M Growth %'] = np.where(
                        df[recent_months[-2]] > 0,
                        ((df[recent_months[-1]] - df[recent_months[-2]]) / df[recent_months[-2]]) * 100,
                        np.where(df[recent_months[-1]] > 0, 100, 0)
                    )
        except Exception as e:
            st.error(f"Error calculating monthly metrics: {str(e)}")
    
    return df, sorted_months

def style_dataframe(df):
    """Apply professional styling to dataframe"""
    # Format all numeric columns appropriately
    format_dict = {}
    for col in df.columns:
        if 'Growth %' in col or 'Margin %' in col or 'Refund%' in col or 'Refund Rate %' in col or 'Sold %' in col:
            format_dict[col] = '{:.1f}%'
        elif 'Revenue' in col or 'Profit' in col or 'COGS' in col or 'Price' in col or 'Landed' in col:
            format_dict[col] = '${:,.2f}'
        elif 'Units' in col or 'Quantity' in col:
            format_dict[col] = '{:,.0f}'
        elif 'Velocity' in col:
            format_dict[col] = '{:.2f}'
    
    # Create styled dataframe
    styler = df.style.format(format_dict)
    
    # Apply conditional formatting
    growth_cols = [col for col in df.columns if 'Growth %' in col]
    for col in growth_cols:
        styler = styler.applymap(lambda x: f"color: {'#28a745' if x > 0 else '#dc3545'}", subset=[col])
    
    if 'Gross Margin %' in df.columns:
        styler = styler.background_gradient(
            subset=['Gross Margin %'],
            cmap='YlGnBu',
            vmin=0,
            vmax=50
        )
    
    if 'Refund Rate %' in df.columns:
        styler = styler.background_gradient(
            subset=['Refund Rate %'],
            cmap='OrRd',
            vmin=0,
            vmax=20
        )
    
    # Highlight important metrics
    highlight_cols = ['Total Revenue', 'Gross Profit', 'Total Units Sold', 'Total COGS']
    for col in highlight_cols:
        if col in df.columns:
            styler = styler.set_properties(
                subset=[col],
                **{'font-weight': 'bold'}
            )
    
    # Set table properties
    styler = styler.set_table_styles([
        {'selector': 'thead th',
         'props': [('background-color', '#2e86ab'), 
                   ('color', 'white'),
                   ('font-weight', 'bold'),
                   ('text-align', 'center')]},
        {'selector': 'tbody tr:nth-child(even)',
         'props': [('background-color', '#f8f9fa')]},
        {'selector': 'tbody tr:hover',
         'props': [('background-color', '#e9ecef')]},
        {'selector': '.total-row',
         'props': [('background-color', '#2e86ab !important'), 
                   ('color', 'white !important'),
                   ('font-weight', 'bold')]}
    ])
    
    return styler

def create_business_performance_table(df):
    """Create business performance table with inventory analysis"""
    if 'Total Unit Ordered' not in df.columns or 'Total Units Sold' not in df.columns:
        return pd.DataFrame()
    
    # Calculate metrics
    total_ordered = df['Total Unit Ordered'].sum()
    total_sold = df['Total Units Sold'].sum()
    sold_pct = (total_sold / total_ordered * 100) if total_ordered > 0 else 0
    remaining_inventory = total_ordered - total_sold
    
    performance_data = {
        'Metric': [
            'Total Inventory Ordered',
            'Total Inventory Sold',
            'Inventory Sold %',
            'Remaining Inventory',
            'Inventory Value at Cost',
            'Potential Revenue',
            'Potential Gross Profit'
        ],
        'Value': [
            total_ordered,
            total_sold,
            sold_pct,
            remaining_inventory,
            remaining_inventory * df['Store Landed'].mean() if 'Store Landed' in df.columns else 0,
            remaining_inventory * df['Retail Price'].mean() if 'Retail Price' in df.columns else 0,
            remaining_inventory * (df['Retail Price'].mean() - df['Store Landed'].mean()) 
            if 'Retail Price' in df.columns and 'Store Landed' in df.columns else 0
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    
    # Format values appropriately
    performance_df.loc[0, 'Value'] = f"{performance_df.loc[0, 'Value']:,.0f}"
    performance_df.loc[1, 'Value'] = f"{performance_df.loc[1, 'Value']:,.0f}"
    performance_df.loc[2, 'Value'] = f"{performance_df.loc[2, 'Value']:.1f}%"
    performance_df.loc[3, 'Value'] = f"{performance_df.loc[3, 'Value']:,.0f}"
    performance_df.loc[4, 'Value'] = f"${performance_df.loc[4, 'Value']:,.2f}"
    performance_df.loc[5, 'Value'] = f"${performance_df.loc[5, 'Value']:,.2f}"
    performance_df.loc[6, 'Value'] = f"${performance_df.loc[6, 'Value']:,.2f}"
    
    return performance_df

def create_product_analysis_table(df, display_cols):
    """Create product analysis table with totals row"""
    # Filter to only include requested columns that exist in the dataframe
    display_cols = [col for col in display_cols if col in df.columns]
    
    if not display_cols:
        return pd.DataFrame()
    
    # Create the main table
    product_table = df[display_cols].copy()
    
    # Add totals row
    totals_row = {}
    for col in display_cols:
        if col in ['Product Name', 'ABO SKU', 'Category']:
            totals_row[col] = 'Grand Total'
        elif '%' in col:
            # For percentage columns, calculate weighted average
            if 'Revenue' in col or 'Profit' in col:
                # Weight by revenue
                if 'Total Revenue' in df.columns:
                    totals_row[col] = (df[col] * df['Total Revenue']).sum() / df['Total Revenue'].sum()
                else:
                    totals_row[col] = df[col].mean()
            else:
                # Simple average for other percentages
                totals_row[col] = df[col].mean()
        elif 'Revenue' in col or 'Profit' in col or 'COGS' in col:
            totals_row[col] = df[col].sum()
        elif 'Units' in col or 'Quantity' in col:
            totals_row[col] = df[col].sum()
        elif 'Velocity' in col:
            totals_row[col] = df[col].mean()
        else:
            totals_row[col] = ''
    
    # Add totals row to the table
    product_table = pd.concat([product_table, pd.DataFrame([totals_row])], ignore_index=True)
    
    return product_table

def create_monthly_breakdown(df, month_cols):
    """Create monthly breakdown table"""
    if not month_cols:
        return pd.DataFrame()
    
    # Calculate monthly metrics
    monthly_data = []
    for month in month_cols:
        month_units = df[month].sum()
        month_revenue = (df[month] * df['Retail Price']).sum() if 'Retail Price' in df.columns else 0
        month_cogs = (df[month] * df['Store Landed']).sum() if 'Store Landed' in df.columns else 0
        month_gp = month_revenue - month_cogs
        month_gm = (month_gp / month_revenue * 100) if month_revenue > 0 else 0
        
        monthly_data.append({
            'Month': month,
            'Units Sold': month_units,
            'Revenue': month_revenue,
            'COGS': month_cogs,
            'Gross Profit': month_gp,
            'Gross Margin %': month_gm
        })
    
    # Calculate totals
    total_units = sum(item['Units Sold'] for item in monthly_data)
    total_revenue = sum(item['Revenue'] for item in monthly_data)
    total_cogs = sum(item['COGS'] for item in monthly_data)
    total_gp = total_revenue - total_cogs
    total_gm = (total_gp / total_revenue * 100) if total_revenue > 0 else 0
    
    # Add totals row
    monthly_data.append({
        'Month': 'Grand Total',
        'Units Sold': total_units,
        'Revenue': total_revenue,
        'COGS': total_cogs,
        'Gross Profit': total_gp,
        'Gross Margin %': total_gm
    })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # Calculate growth rates if we have at least 2 months
    if len(month_cols) >= 2:
        for i in range(1, len(month_cols)):
            current_month = month_cols[i]
            prev_month = month_cols[i-1]
            
            # Find the corresponding rows in our monthly data
            current_idx = next((idx for idx, item in enumerate(monthly_data) if item['Month'] == current_month), None)
            prev_idx = next((idx for idx, item in enumerate(monthly_data) if item['Month'] == prev_month), None)
            
            if current_idx is not None and prev_idx is not None:
                # Units growth
                if monthly_data[prev_idx]['Units Sold'] > 0:
                    units_growth = ((monthly_data[current_idx]['Units Sold'] - monthly_data[prev_idx]['Units Sold']) / 
                                   monthly_data[prev_idx]['Units Sold']) * 100
                else:
                    units_growth = 100 if monthly_data[current_idx]['Units Sold'] > 0 else 0
                
                # Revenue growth
                if monthly_data[prev_idx]['Revenue'] > 0:
                    revenue_growth = ((monthly_data[current_idx]['Revenue'] - monthly_data[prev_idx]['Revenue']) / 
                                     monthly_data[prev_idx]['Revenue']) * 100
                else:
                    revenue_growth = 100 if monthly_data[current_idx]['Revenue'] > 0 else 0
                
                # Add to dataframe
                monthly_df.loc[current_idx, f'{current_month} Units Growth %'] = units_growth
                monthly_df.loc[current_idx, f'{current_month} Revenue Growth %'] = revenue_growth
    
    return monthly_df

def create_sku_trend_table(df, month_cols, selected_months):
    """Create SKU performance trend table with priority indicators"""
    if not month_cols or not selected_months:
        return pd.DataFrame()
    
    # Calculate total sales for selected months
    df['Selected Period Sales'] = df[selected_months].sum(axis=1)
    
    # Sort by sales volume
    trend_df = df.sort_values('Selected Period Sales', ascending=False)
    
    # Add priority column
    bins = [0, 0.33, 0.66, 1]
    labels = ['High', 'Medium', 'Low']
    trend_df['Priority'] = pd.cut(trend_df['Selected Period Sales'].rank(pct=True), bins=bins, labels=labels)
    
    # Select columns to display
    display_cols = ['ABO SKU', 'Product Name', 'Selected Period Sales', 'Priority']
    if 'Category' in trend_df.columns:
        display_cols.insert(2, 'Category')
    
    return trend_df[display_cols]

def style_sku_trend_table(df):
    """Apply special styling to SKU trend table"""
    # Format numbers
    styler = df.style.format({
        'Selected Period Sales': '{:,.0f}'
    })
    
    # Apply priority coloring
    styler = styler.applymap(lambda x: 'background-color: #ffcccc' if x == 'High' else 
                                      'background-color: #fff3cd' if x == 'Medium' else 
                                      'background-color: #d4edda',
                            subset=['Priority'])
    
    # Set table properties
    styler = styler.set_table_styles([
        {'selector': 'thead th',
         'props': [('background-color', '#2e86ab'), 
                   ('color', 'white'),
                   ('font-weight', 'bold'),
                   ('text-align', 'center')]},
        {'selector': 'tbody tr:nth-child(even)',
         'props': [('background-color', '#f8f9fa')]},
        {'selector': 'tbody tr:hover',
         'props': [('background-color', '#e9ecef')]},
        {'selector': '.total-row',
         'props': [('background-color', '#2e86ab !important'), 
                   ('color', 'white !important'),
                   ('font-weight', 'bold')]}
    ])
    
    return styler

def load_default_data():
    """Load default data.csv file if it exists"""
    if os.path.exists('data.csv'):
        try:
            # First try UTF-8
            try:
                return pd.read_csv('data.csv')
            except UnicodeDecodeError:
                # If UTF-8 fails, detect encoding
                with open('data.csv', 'rb') as f:
                    rawdata = f.read()
                    result = chardet.detect(rawdata)
                    encoding = result['encoding']
                
                # Try detected encoding with error handling
                try:
                    return pd.read_csv('data.csv', encoding=encoding)
                except:
                    # If detected encoding fails, try common alternatives
                    for enc in ['latin1', 'iso-8859-1', 'cp1252']:
                        try:
                            return pd.read_csv('data.csv', encoding=enc)
                        except:
                            continue
        except Exception as e:
            st.error(f"Error loading default data file: {str(e)}")
    return None

def main():
    # Check for default data file
    default_df = load_default_data()
    
    # File upload with default option
    if default_df is not None:
        st.info("Default data.csv file found in the same directory. You can use this or upload your own file.")
        use_default = st.checkbox("Use default data.csv file", value=True)
    else:
        use_default = False
    
    if use_default:
        df = default_df.copy()
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=["csv"], help="Upload your sales data with monthly columns like 24-Oct, 25-Jan")
        if uploaded_file is not None:
            try:
                # First try UTF-8
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    # If UTF-8 fails, detect encoding
                    uploaded_file.seek(0)  # Reset file pointer
                    rawdata = uploaded_file.read()
                    result = chardet.detect(rawdata)
                    encoding = result['encoding']
                    uploaded_file.seek(0)  # Reset file pointer again
                    
                    # Try detected encoding with error handling
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                    except:
                        # If detected encoding fails, try common alternatives
                        for enc in ['latin1', 'iso-8859-1', 'cp1252']:
                            try:
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file, encoding=enc)
                                break
                            except:
                                continue
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                return
    
    if ('df' in locals() and df is not None) or (use_default and default_df is not None):
        month_cols = detect_month_columns(df)
        
        if not month_cols:
            st.error("No month columns detected. Please ensure your data has columns like '24-Oct', '25-Jan'.")
            return

        # Sidebar filters
        with st.sidebar:
            st.header("Analysis Settings")
            
            # Month selection
            selected_months = st.multiselect(
                "Select Months to Analyze",
                options=month_cols,
                default=month_cols,
                help="Select specific months to include in analysis"
            )
            
            # Category filter
            if 'Category' in df.columns:
                categories = ['All'] + sorted(df['Category'].unique().tolist())
                selected_category = st.selectbox(
                    "Filter by Product Category",
                    options=categories,
                    index=0,
                    help="Select product category to analyze"
                )
                
                # Product Name filter based on selected category
                if selected_category != 'All':
                    products_in_category = ['All'] + sorted(df[df['Category'] == selected_category]['Product Name'].unique().tolist())
                else:
                    products_in_category = ['All'] + sorted(df['Product Name'].unique().tolist())
                
                selected_products = st.multiselect(
                    "Filter by Product Name",
                    options=products_in_category,
                    default=['All'],
                    help="Select specific products to analyze"
                )
            else:
                selected_category = 'All'
                selected_products = ['All']
            
            # Display options
            st.markdown("**Display Options**")
            show_monthly_breakdown = st.checkbox("Show Monthly Breakdown", value=True)
            show_growth_metrics = st.checkbox("Show Growth Metrics", value=True)
            show_additional_charts = st.checkbox("Show Additional Charts", value=True)
        
        # Calculate metrics for selected months
        df, sorted_months = calculate_all_metrics(df, month_cols, selected_months)
        
        # Apply filters
        if 'Category' in df.columns and selected_category != 'All':
            filtered_df = df[df['Category'] == selected_category].copy()
        else:
            filtered_df = df.copy()
        
        if 'Product Name' in df.columns and selected_products and 'All' not in selected_products:
            filtered_df = filtered_df[filtered_df['Product Name'].isin(selected_products)].copy()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Performance Summary", "Product Analysis", "Monthly Breakdown", "SKU Performance Trend"])
        
        with tab1:
            st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)
            
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Products", len(filtered_df))
            
            if 'Total Revenue' in filtered_df.columns:
                col2.metric("Total Revenue", f"${filtered_df['Total Revenue'].sum():,.0f}")
            if 'Gross Profit' in filtered_df.columns:
                col3.metric("Gross Profit", f"${filtered_df['Gross Profit'].sum():,.0f}")
            if 'Gross Margin %' in filtered_df.columns:
                col4.metric("Avg Gross Margin", f"{filtered_df['Gross Margin %'].mean():.1f}%")
            
            # Cumulative Performance Table
            st.markdown('<p class="section-header">Cumulative Performance</p>', unsafe_allow_html=True)
            
            summary_data = {
                'Metric': ['Total Units Ordered', 'Total Units Sold'],
                'Value': [
                    filtered_df['Total Unit Ordered'].sum() if 'Total Unit Ordered' in filtered_df.columns else 0,
                    filtered_df['Total Units Sold'].sum()
                ]
            }
            
            if 'Total Revenue' in filtered_df.columns:
                summary_data['Metric'].append('Total Revenue')
                summary_data['Value'].append(filtered_df['Total Revenue'].sum())
            
            if 'Gross Profit' in filtered_df.columns:
                summary_data['Metric'].append('Gross Profit')
                summary_data['Value'].append(filtered_df['Gross Profit'].sum())
            
            if 'Gross Margin %' in filtered_df.columns:
                summary_data['Metric'].append('Avg Gross Margin %')
                summary_data['Value'].append(filtered_df['Gross Margin %'].mean())
            
            summary_df = pd.DataFrame(summary_data)
            
            # Format values
            summary_df['Value'] = summary_df.apply(lambda x: 
                f"${x['Value']:,.0f}" if x['Metric'] in ['Total Revenue', 'Gross Profit'] else
                f"{x['Value']:,.0f}" if x['Metric'] in ['Total Units Ordered', 'Total Units Sold'] else
                f"{x['Value']:.1f}%", axis=1)
            
            st.dataframe(
                style_dataframe(summary_df),
                use_container_width=True,
                height=200
            )
            
            # Business Performance Table
            st.markdown('<p class="section-header">Business Performance</p>', unsafe_allow_html=True)
            business_perf = create_business_performance_table(filtered_df)
            if not business_perf.empty:
                st.dataframe(
                    business_perf,
                    use_container_width=True,
                    height=300
                )
            else:
                st.warning("Required columns ('Total Unit Ordered' and 'Total Units Sold') not found for business performance analysis")
        
        with tab2:
            st.markdown('<p class="section-header">Product Analysis</p>', unsafe_allow_html=True)
            
            # Product selection for detailed view with "All" option
            if 'Product Name' in filtered_df.columns:
                product_list = ['All'] + filtered_df['Product Name'].unique().tolist()
                selected_product = st.selectbox(
                    "Select Product for Detailed View",
                    options=product_list,
                    index=0,
                    help="Select a specific product to view detailed metrics or 'All' to see all products"
                )
                
                # Display selected product details or all if "All" selected
                if selected_product != 'All':
                    product_data = filtered_df[filtered_df['Product Name'] == selected_product]
                    if not product_data.empty:
                        st.markdown(f"**Detailed View for Product: {selected_product}**")
                        st.dataframe(
                            style_dataframe(product_data),
                            use_container_width=True
                        )
            
            # Determine which columns to show in main table
            display_cols = ['Product Name']
            if 'ABO SKU' in filtered_df.columns:
                display_cols.insert(0, 'ABO SKU')
            if 'Category' in filtered_df.columns:
                display_cols.insert(1, 'Category')
            
            display_cols.extend([
                'Total Units Sold', 
                'Total Revenue',
                'Total COGS',
                'Gross Profit',
                'Gross Margin %'
            ])
            
            if show_growth_metrics and '3M Growth %' in filtered_df.columns:
                display_cols.append('3M Growth %')
            if 'Sales Velocity (units/day)' in filtered_df.columns:
                display_cols.append('Sales Velocity (units/day)')
            if 'Refund Rate %' in filtered_df.columns:
                display_cols.append('Refund Rate %')
            
            # Create table with totals
            product_table = create_product_analysis_table(filtered_df, display_cols)
            
            # Sort options
            sort_options = [col for col in display_cols if col not in ['ABO SKU', 'Product Name', 'Category']]
            default_sort = 'Total Revenue' if 'Total Revenue' in sort_options else sort_options[0] if sort_options else None
            
            if sort_options:
                sort_by = st.selectbox(
                    "Sort By",
                    options=sort_options,
                    index=sort_options.index(default_sort) if default_sort else 0
                )
                
                # Sort the table (excluding the totals row)
                sorted_table = pd.concat([
                    product_table[product_table['ABO SKU'] != 'Grand Total'].sort_values(
                        by=sort_by, 
                        ascending=False
                    ),
                    product_table[product_table['ABO SKU'] == 'Grand Total']
                ])
                
                # Display sorted data
                st.dataframe(
                    style_dataframe(sorted_table),
                    use_container_width=True,
                    height=600
                )
            else:
                st.warning("No numeric columns available for sorting")
            
            # Additional charts section
            if show_additional_charts:
                st.markdown('<p class="section-header">Additional Product Metrics</p>', unsafe_allow_html=True)
                
                # Top 10 Products by Revenue
                st.markdown("**Top 10 Products by Revenue**")
                top_products = filtered_df.nlargest(10, 'Total Revenue')[['Product Name', 'Total Revenue']]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(top_products['Product Name'], top_products['Total Revenue'], color='#2e86ab')
                ax.set_xlabel('Total Revenue ($)')
                ax.set_ylabel('Product Name')
                st.pyplot(fig)
                
                # Gross Margin Distribution
                st.markdown("**Gross Margin Distribution**")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(filtered_df['Gross Margin %'], bins=20, color='#2e86ab', edgecolor='black')
                ax.set_xlabel('Gross Margin %')
                ax.set_ylabel('Number of Products')
                st.pyplot(fig)
        
        with tab3:
            if show_monthly_breakdown:
                st.markdown('<p class="section-header">Monthly Performance Breakdown</p>', unsafe_allow_html=True)
                
                monthly_breakdown = create_monthly_breakdown(filtered_df, selected_months)
                if not monthly_breakdown.empty:
                    # Determine which columns to show
                    breakdown_cols = ['Month', 'Units Sold', 'Revenue', 'COGS', 'Gross Profit', 'Gross Margin %']
                    if show_growth_metrics:
                        growth_cols = [col for col in monthly_breakdown.columns if 'Growth %' in col]
                        breakdown_cols.extend(growth_cols)
                    
                    # Apply styling
                    styled_breakdown = style_dataframe(monthly_breakdown[breakdown_cols])
                    
                    # Highlight the total row
                    total_row_index = monthly_breakdown[monthly_breakdown['Month'] == 'Grand Total'].index[0]
                    styled_breakdown = styled_breakdown.set_table_styles([
                        {'selector': f'.row{total_row_index}',
                         'props': [('background-color', '#2e86ab'), 
                                   ('color', 'white'),
                                   ('font-weight', 'bold')]}
                    ], overwrite=False)
                    
                    st.dataframe(
                        styled_breakdown,
                        use_container_width=True,
                        height=600
                    )
                    
                    # Monthly trends visualization
                    st.markdown('<p class="section-header">Monthly Trends</p>', unsafe_allow_html=True)
                    trend_options = [col for col in ['Units Sold', 'Revenue', 'Gross Profit', 'Gross Margin %'] if col in monthly_breakdown.columns]
                    if trend_options:
                        trend_metric = st.selectbox(
                            "View Trend For",
                            options=trend_options,
                            index=0
                        )
                        
                        # Exclude total row from chart and ensure correct order
                        chart_data = monthly_breakdown[monthly_breakdown['Month'] != 'Grand Total']
                        try:
                            # Convert month format from 24-Oct to Oct-24 for sorting
                            chart_data['Month'] = chart_data['Month'].apply(lambda x: f"{x.split('-')[1]}-{x.split('-')[0]}")
                            chart_data['Month'] = pd.to_datetime(chart_data['Month'], format='%b-%y')
                            chart_data = chart_data.sort_values('Month')
                            chart_data['Month'] = chart_data['Month'].dt.strftime('%b-%y')
                        except:
                            pass  # If month format can't be parsed, use as-is
                        
                        st.line_chart(
                            chart_data.set_index('Month')[trend_metric],
                            use_container_width=True,
                            height=400
                        )
                        
                        # Additional monthly charts
                        if show_additional_charts:
                            st.markdown('<p class="section-header">Additional Monthly Metrics</p>', unsafe_allow_html=True)
                            
                            # Monthly Revenue vs COGS
                            st.markdown("**Monthly Revenue vs COGS**")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(chart_data['Month'], chart_data['Revenue'], label='Revenue', marker='o')
                            ax.plot(chart_data['Month'], chart_data['COGS'], label='COGS', marker='o')
                            ax.set_xlabel('Month')
                            ax.set_ylabel('Amount ($)')
                            ax.legend()
                            st.pyplot(fig)
                            
                            # Monthly Gross Margin Trend
                            st.markdown("**Monthly Gross Margin Trend**")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(chart_data['Month'], chart_data['Gross Margin %'], label='Gross Margin %', marker='o', color='green')
                            ax.set_xlabel('Month')
                            ax.set_ylabel('Gross Margin %')
                            ax.axhline(y=chart_data['Gross Margin %'].mean(), color='r', linestyle='--', label='Average')
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        st.warning("No trend data available")
                else:
                    st.warning("No monthly breakdown data available")
        
        with tab4:
            st.markdown('<p class="section-header">SKU Performance Trend</p>', unsafe_allow_html=True)
            
            sku_trend = create_sku_trend_table(filtered_df, month_cols, selected_months)
            if not sku_trend.empty:
                st.dataframe(
                    style_sku_trend_table(sku_trend),
                    use_container_width=True,
                    height=600
                )
            else:
                st.warning("No data available for SKU performance trend")
        
        # Export buttons
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Export Data**")
        
        if st.sidebar.button("Export Current View"):
            # Prepare data for export based on active tab
            if tab1.active:
                export_data = filtered_df
            elif tab2.active:
                export_data = product_table
            elif tab3.active:
                export_data = monthly_breakdown if show_monthly_breakdown else pd.DataFrame()
            else:
                export_data = sku_trend
            
            if not export_data.empty:
                csv = export_data.to_csv(index=False).encode('utf-8')
                st.sidebar.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="sales_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.sidebar.warning("No data to export")
    
    else:
        st.info("Please upload a CSV file to begin analysis or ensure data.csv exists in the same directory")
        st.markdown("""
        ### Expected Data Format:
        The system will automatically detect and analyze:
        - **Product identifiers**: Product Name, ABO SKU, Category
        - **Monthly sales columns**: Format like 24-Oct, 25-Jan
        - **Pricing data**: Retail Price, Store Landed Cost
        - **Performance metrics**: Total Unit Ordered, Sold %, Quantity Refunded
        
        *For best results, save your file as UTF-8 encoded CSV.*
        """)

if __name__ == "__main__":
    main()