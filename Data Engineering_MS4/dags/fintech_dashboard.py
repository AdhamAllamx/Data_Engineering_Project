import streamlit as st
import pandas as pd
import plotly.express as px

def run_dashboard(filename):
    df = pd.read_csv(filename)
    
    # Title
    st.title("FinTech Dashboard")
    st.subheader("Created by Adham Allam :), ID: 52_24625")


    st.header("1. Distribution of Loan Amounts by Grade")

    grade_order = sorted(df["letter_grade"].unique()) 

    fig1 = px.box(df,x="letter_grade",y="loan_amount",title="Loan Amount Distribution by Grade",labels={"letter_grade": "Loan Grade", "loan_amount": "Loan Amount"},
    color="letter_grade",category_orders={"letter_grade": grade_order} )
    st.plotly_chart(fig1)


    st.header("2. Loan Amount vs Annual Income by State")
    state_filter = st.selectbox("Select State", ["All"] + sorted(df['state'].unique()))
    filtered_df = df if state_filter == "All" else df[df['state'] == state_filter]
    fig2 = px.scatter(filtered_df, x="annual_inc", y="loan_amount", color="loan_status",title="Loan Amount vs Annual Income")
    st.plotly_chart(fig2)

    st.header("3. Loan Issuance Trend")
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['year'] = df['issue_date'].dt.year
    available_years = ["All"] + sorted(df['year'].unique().tolist())
    st.subheader("Interactive Line Graph by Month and Year")
    selected_year = st.selectbox("Select Year", available_years)
    trend_fig = create_trend_line_graph(df, selected_year)
    st.plotly_chart(trend_fig)
    
    # Question 4: States with Highest Average Loan Amount


    st.header("4. States with Highest Average Loan Amount")

    choropleth_fig = create_choropleth_map(df)
    st.subheader("4.1 Interactive Choropleth Map")
    st.plotly_chart(choropleth_fig)

    st.subheader("4.2 States with Highest Average Loan Amount")

    avg_loan_by_state = df.groupby('state')['loan_amount'].mean().sort_values(ascending=False)
    st.bar_chart(avg_loan_by_state.head(10))




    # Question 5: Loan Grade Distribution
    st.header("5. Percentage Distribution of Loan Grades")
    grade_dist = df['letter_grade'].value_counts(normalize=True) * 100
    st.bar_chart(grade_dist)
   

def create_choropleth_map(df):

    state_avg_loan = df.groupby('state', as_index=False)['loan_amount'].mean()

    state_avg_loan.columns = ['state', 'average_loan_amount']

    fig = px.choropleth(
        state_avg_loan,
        locations="state",            
        locationmode="USA-states",   
        color="average_loan_amount",  
        color_continuous_scale="Blues",  
        scope="usa",                 
        labels={'average_loan_amount': 'Avg Loan Amount'},
        title="Average Loan Amount by State",
    )

    fig.update_layout(
        geo=dict( lakecolor="rgb(255, 255, 255)"   ),title_x=0.5,)

    return fig


def create_trend_line_graph(df, selected_year):

    df['issue_date'] = pd.to_datetime(df['issue_date'])

    # Extract year and month for grouping
    df['year'] = df['issue_date'].dt.year
    df['month'] = df['issue_date'].dt.month

    # Filter the data based on the selected year
    if selected_year != "All":
        df = df[df['year'] == int(selected_year)]

    trend_data = df.groupby(['month', 'year'], as_index=False).agg(
        loan_count=('loan_amount', 'count'),    
        total_loan_amount=('loan_amount', 'sum') )

    fig = px.line(
        trend_data,
        x="month",
        y="loan_count",  
        title="Trend of Loan Issuance Over the Months",
        color="year",    
        labels={'month': 'Month', 'loan_count': 'Number of Loans'},
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),  
        title_x=0.5,  
    )

    return fig



if __name__ == "__main__":
    run_dashboard('/opt/airflow/data/fintech_transformed.csv')
