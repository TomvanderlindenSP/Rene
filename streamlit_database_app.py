import streamlit as st
import openai
import pandas as pd
from sqlalchemy import create_engine, text
import re
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Product Sales Query Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #566573;
        margin-bottom: 2rem;
    }
    .question-input {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #F0F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86C1;
        margin: 1rem 0;
    }
    .sql-display {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    .example-card {
        background-color: #FEF9E7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F39C12;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .example-card:hover {
        background-color: #FCF3CF;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    .error-box {
        background-color: #FADBD8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E74C3C;
        color: #C0392B;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D5EDDA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28A745;
        color: #155724;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleProductQuery:
    def __init__(self, openai_api_key, database_url):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.engine = create_engine(database_url)
        self.schema_info = {}
        
    def analyze_database(self):
        """Get database schema"""
        try:
            with self.engine.connect() as conn:
                # Get all tables
                tables_query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """
                tables_df = pd.read_sql(text(tables_query), conn)
                
                # Get columns for each table
                for table_name in tables_df['table_name']:
                    columns_query = f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        AND table_schema = 'public'
                        ORDER BY ordinal_position;
                    """
                    columns_df = pd.read_sql(text(columns_query), conn)
                    self.schema_info[table_name] = columns_df.to_dict('records')
                
                return True, f"✅ Connected! Found {len(tables_df)} tables"
        except Exception as e:
            return False, f"❌ Connection failed: {e}"
    
    def create_schema_context(self):
        """Create database context for AI"""
        context = "DATABASE SCHEMA:\n\n"
        
        for table_name, columns in self.schema_info.items():
            context += f"TABLE: {table_name}\n"
            for col in columns:
                context += f"  - {col['column_name']} ({col['data_type']})\n"
            context += "\n"
        
        context += """
QUERY GUIDELINES:
- Focus on product sales, quantities, counts, totals
- Use JOINs to connect related tables
- Add LIMIT for "top" queries (default 10)
- Use ILIKE for text searches
- Group and aggregate data appropriately
- Order results logically
- For ilike Always use upper to avoid problems on the caps

EVENT-SPECIFIC INSTRUCTIONS:
- ALWAYS use the field 'internal_name' when querying for event-related information
- When asked about specific events (e.g., "pl25", "pl 25"), match against internal_name field
- For event queries, internal_name corresponds to the event identifier

TICKET COUNTING INSTRUCTIONS:
- For paid tickets: Use COUNT(DISTINCT id) FROM paid_ticket_tables GROUP BY internal_name
- For free tickets: Use free_tickets table
- For delegates/sponsorship tickets: Use delegates table
- Each ticket type has its own dedicated table

CONTENT DOWNLOAD INSTRUCTIONS:
- For questions about content downloads: Query the metabase_resource_downloads table
- Downloads data structure: count(resource_id) as downloads, grouped by resource_id and resource_type
- For date filtering: use datetime column (format: datetime >= '2024-01-01')
- For averages by type: use subquery pattern to first count downloads per resource, then average by resource_type
        """
        return context
    
    def generate_sql(self, question):
        """Convert question to SQL"""
        schema_context = self.create_schema_context()
        
        prompt = f"""
Convert this question into a PostgreSQL query using the provided schema.

{schema_context}

RULES:
1. Return ONLY valid PostgreSQL SQL
2. No explanations, just the query
3. Use proper JOINs between tables
4. Add LIMIT for top/best queries
5. Use ILIKE for text searches

SPECIAL INSTRUCTIONS:
- For EVENT queries: Always filter by 'internal_name' field (e.g., internal_name = 'pl25')
- For TICKET COUNTS: Use COUNT(DISTINCT id) grouped by internal_name
  * Paid tickets: query paid_ticket_tables
  * Free tickets: query free_tickets table  
  * Delegates/sponsorship: query delegates table
- For CONTENT DOWNLOADS: Query metabase_ resource_downloads table

QUESTION: {question}

SQL:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Return only valid PostgreSQL queries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0
            )
            
            sql = response.choices[0].message.content.strip()
            # Clean up SQL
            sql = re.sub(r'```sql\n?', '', sql)
            sql = re.sub(r'\n?```', '', sql)
            sql = sql.strip()
            
            return sql
        except Exception as e:
            return f"Error generating SQL: {e}"
    
    def execute_query(self, sql):
        """Execute SQL query"""
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
                return df, None
        except Exception as e:
            return None, f"Query error: {e}"
    
    def explain_results(self, question, df):
        """Generate explanation of results"""
        if df is None or len(df) == 0:
            return "No results found for your question."
        
        # Create summary
        summary = f"Found {len(df)} result(s) for your question: '{question}'\n\n"
        
        # Add key insights based on data
        if len(df) == 1 and len(df.columns) == 1:
            # Single value result
            value = df.iloc[0, 0]
            summary += f"Answer: {value:,}" if isinstance(value, (int, float)) else f"Answer: {value}"
        else:
            # Multiple results
            summary += f"Results show {len(df)} rows with {len(df.columns)} columns."
            
            # Highlight numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:2]:  # Show first 2 numeric columns
                    total = df[col].sum()
                    avg = df[col].mean()
                    summary += f"\n• {col}: Total = {total:,.0f}, Average = {avg:.1f}"
        
        return summary

def initialize_session_state():
    """Initialize session state"""
    if 'query_tool' not in st.session_state:
        st.session_state.query_tool = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Product Sales Query Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask questions about your product sales data in plain English</p>', unsafe_allow_html=True)
    
    # Configuration sidebar
    with st.sidebar:
        st.header("🔧 Setup")
        
        openai_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
        database_url = st.text_input("Database URL", type="password", help="PostgreSQL connection string")
        
        if st.button("🔌 Connect", use_container_width=True):
            if openai_key and database_url:
                with st.spinner("Connecting to database..."):
                    try:
                        query_tool = SimpleProductQuery(openai_key, database_url)
                        success, message = query_tool.analyze_database()
                        
                        if success:
                            st.session_state.query_tool = query_tool
                            st.session_state.connected = True
                            st.success(message)
                        else:
                            st.error(message)
                    except Exception as e:
                        st.error(f"Setup failed: {e}")
            else:
                st.error("Please provide both API key and database URL")
        
        # Show database info if connected
        if st.session_state.connected:
            st.success("✅ Connected!")
            
            with st.expander("📋 Database Tables"):
                for table_name in st.session_state.query_tool.schema_info.keys():
                    st.write(f"• **{table_name}**")
                    for col in st.session_state.query_tool.schema_info[table_name][:5]:
                        st.write(f"  - {col['column_name']}")
                    if len(st.session_state.query_tool.schema_info[table_name]) > 5:
                        st.write(f"  ... and {len(st.session_state.query_tool.schema_info[table_name]) - 5} more")
    
    # Main interface
    if not st.session_state.connected:
        st.info("👈 Please connect to your database using the sidebar")
        
        # Show example questions
        st.subheader("💡 Example Questions")
        examples = [
            "How many units of product X have been sold?",
            "What are the top 10 best selling products?",
            "Show me total sales by product category",
            "How many customers bought product Y?",
            "What's the total revenue for each product?",
            "Which products were sold in the last 30 days?",
            "Show me products with sales greater than 100 units"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                st.markdown(f'<div class="example-card">📝 {example}</div>', unsafe_allow_html=True)
        
        return
    
    # Query interface
    st.subheader("🤔 Ask Your Question")
    
    # Quick question buttons
    st.write("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        "Total products sold",
        "Best selling products",
        "Sales by category", 
        "Recent sales"
    ]
    
    selected_quick = None
    with col1:
        if st.button(quick_questions[0]):
            selected_quick = "How many total products have been sold?"
    with col2:
        if st.button(quick_questions[1]):
            selected_quick = "What are the top 10 best selling products?"
    with col3:
        if st.button(quick_questions[2]):
            selected_quick = "Show me total sales by product category"
    with col4:
        if st.button(quick_questions[3]):
            selected_quick = "What products were sold in the last 7 days?"
    
    # Text input for custom questions
    question = st.text_input(
        "Or type your own question:",
        value=selected_quick if selected_quick else "",
        placeholder="e.g., How many units of iPhone have been sold?"
    )
    
    # Process question
    if st.button("🔍 Get Answer", use_container_width=True) and question:
        with st.spinner("🤖 Generating SQL and fetching data..."):
            try:
                # Generate SQL
                sql = st.session_state.query_tool.generate_sql(question)
                
                # Show generated SQL
                st.subheader("🔧 Generated SQL")
                st.markdown(f'<div class="sql-display">{sql}</div>', unsafe_allow_html=True)
                
                # Execute query
                df, error = st.session_state.query_tool.execute_query(sql)
                
                if error:
                    st.markdown(f'<div class="error-box">❌ {error}</div>', unsafe_allow_html=True)
                else:
                    # Show results
                    st.subheader("📊 Results")
                    
                    # Summary
                    explanation = st.session_state.query_tool.explain_results(question, df)
                    st.markdown(f'<div class="success-box">{explanation}</div>', unsafe_allow_html=True)
                    
                    # Data table
                    st.subheader("📋 Data")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Export options
                    st.subheader("📥 Export Data")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV",
                            data=csv,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        excel_buffer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
                        df.to_excel(excel_buffer, index=False, sheet_name='Results')
                        excel_buffer.close()
                        excel_buffer.seek(0)
                        
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer.read(),
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col3:
                        json_data = df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="🔧 Download JSON",
                            data=json_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Quick stats
                    if len(df) > 0:
                        st.subheader("📈 Quick Stats")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f'<div class="metric-box"><h3>{len(df):,}</h3><p>Total Rows</p></div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'<div class="metric-box"><h3>{len(df.columns)}</h3><p>Columns</p></div>', unsafe_allow_html=True)
                        
                        with col3:
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                total_sum = df[numeric_cols[0]].sum()
                                st.markdown(f'<div class="metric-box"><h3>{total_sum:,.0f}</h3><p>Sum of {numeric_cols[0]}</p></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="metric-box"><h3>-</h3><p>No numeric data</p></div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
