
import os
import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq


# =====================================================
# 1. APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SmartSupport AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

MODEL = "openai/gpt-oss-120b"


# =====================================================
# 2. RESPONSIVE CSS - DESKTOP + ANDROID
# =====================================================

st.markdown("""
<style>

/* ---------- MAIN APPLICATION ---------- */

.stApp {
    background: #f3f6fc !important;
    color: #172554 !important;
}

[data-testid="stAppViewContainer"] {
    background: #f3f6fc !important;
}

[data-testid="stMain"] {
    background: #f3f6fc !important;
}

[data-testid="stHeader"] {
    background: #f3f6fc !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 5rem;
}

/* ---------- HEADINGS ---------- */

h1, h2, h3, h4 {
    color: #174ea6 !important;
    font-weight: 800 !important;
}

p, li, label, .stMarkdown {
    color: #243b63;
}

/* ---------- CHAT MESSAGE CARDS ---------- */

div[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #d8e3f5 !important;
    border-radius: 16px !important;
    padding: 14px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 3px 12px rgba(23, 78, 166, 0.07);
}

/* FORCE DARK TEXT INSIDE ALL CHAT MESSAGES */

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em,
div[data-testid="stChatMessage"] .stMarkdown,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #172554 !important;
}

/* Chat headings */

div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4 {
    color: #174ea6 !important;
}

/* ---------- CHAT INPUT ---------- */

div[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 2px solid #b7cef5 !important;
    border-radius: 15px !important;
}

div[data-testid="stChatInput"] textarea {
    color: #172554 !important;
    background: #ffffff !important;
    font-size: 16px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* ---------- BUTTONS ---------- */

.stButton button,
.stDownloadButton button,
.stFormSubmitButton button {
    background: #1769e0 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    padding: 9px 18px !important;
    min-height: 42px !important;
}

.stButton button p,
.stDownloadButton button p,
.stFormSubmitButton button p {
    color: #ffffff !important;
}

.stButton button:hover {
    background: #0d47a1 !important;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: #eaf1ff !important;
}

[data-testid="stSidebar"] * {
    color: #173b75 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: #174ea6 !important;
}

/* ---------- INPUT FIELDS ---------- */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    color: #172554 !important;
    background: #ffffff !important;
    border: 1px solid #b7cef5 !important;
    border-radius: 10px !important;
    font-size: 16px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* ---------- INFO / SUCCESS / WARNING ---------- */

[data-testid="stAlert"] {
    border-radius: 12px !important;
}

/* ---------- DATA TABLES ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #d8e3f5;
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- MOBILE RESPONSIVE DESIGN ---------- */

@media screen and (max-width: 768px) {

    [data-testid="stMainBlockContainer"] {
        padding-left: 12px !important;
        padding-right: 12px !important;
        padding-top: 1rem !important;
    }

    h1 {
        font-size: 27px !important;
    }

    h2 {
        font-size: 22px !important;
    }

    h3 {
        font-size: 19px !important;
    }

    div[data-testid="stChatMessage"] {
        padding: 11px !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
    }

    div[data-testid="stChatInput"] textarea {
        font-size: 16px !important;
    }

    .stButton button {
        width: 100%;
        min-height: 44px !important;
    }

    [data-testid="stSidebar"] {
        min-width: 250px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. GROQ API KEY
# =====================================================

def get_api_key():
    try:
        return st.secrets.get(
            "GROQ_API_KEY",
            os.getenv("GROQ_API_KEY", "")
        )
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

client = Groq(api_key=api_key) if api_key else None


# =====================================================
# 4. CSV DATABASE LOADING
# =====================================================

def load_data(filename, default_columns):

    if not os.path.exists(filename):
        return pd.DataFrame(columns=default_columns)

    try:
        df = pd.read_csv(
            filename,
            dtype=str,
            encoding="utf-8-sig"
        )

        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        df = df.fillna("")

        for column in df.columns:
            df[column] = df[column].astype(str).str.strip()

        return df

    except Exception as error:
        st.error(f"Error loading {filename}: {error}")
        return pd.DataFrame(columns=default_columns)


customers = load_data(
    "customers.csv",
    ["customer_id", "name", "email", "phone"]
)

orders = load_data(
    "orders.csv",
    ["order_id", "customer_name", "phone", "product", "order_date"]
)

tickets = load_data(
    "tickets.csv",
    ["ticket_id", "customer_id", "subject", "message",
     "status", "created_at"]
)

knowledge = load_data(
    "knowledge_base.csv",
    ["question", "answer"]
)


# =====================================================
# 5. ORDER ID NORMALIZATION
# =====================================================

def normalize_order_id(order_id):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(order_id).strip().upper()
    )


def extract_order_id(message):

    pattern = r"\bORD(?:[-\s]?\d{4})?[-\s]?\d{3,}\b"

    match = re.search(
        pattern,
        str(message).upper()
    )

    if match:
        return match.group(0).strip()

    return None


# =====================================================
# 6. SEARCH ORDER DATABASE
# =====================================================

def search_order(order_id):

    if orders.empty:
        return None

    possible_columns = [
        "order_id",
        "orderid",
        "order_number",
        "order_no"
    ]

    order_column = next(
        (
            column
            for column in possible_columns
            if column in orders.columns
        ),
        None
    )

    if order_column is None:
        return None

    searched_id = normalize_order_id(order_id)

    for _, order in orders.iterrows():

        database_id = normalize_order_id(
            order.get(order_column, "")
        )

        if database_id == searched_id:
            return order.to_dict()

    return None


# =====================================================
# 7. FORMAT ORDER DETAILS
# =====================================================

def format_order_response(order):

    response = "### 📦 Your Order Details\n\n"

    fields = {
        "order_id": "Order ID",
        "customer_name": "Customer Name",
        "phone": "Phone Number",
        "product": "Product",
        "order_date": "Order Date",
        "status": "Order Status",
        "amount": "Order Amount",
        "tracking_number": "Tracking Number",
        "expected_delivery": "Expected Delivery"
    }

    for key, label in fields.items():

        value = order.get(key, "")

        if value and str(value).strip():
            response += f"**{label}:** {value}\n\n"

    if not any(order.get(key, "") for key in fields):
        response += "Order found, but no details are available."

    response += "\nHow else can I help you today?"

    return response


# =====================================================
# 8. CUSTOMER LOOKUP
# =====================================================

def find_customer(customer_id):

    if customers.empty:
        return None

    if "customer_id" not in customers.columns:
        return None

    result = customers[
        customers["customer_id"].astype(str).str.strip().str.upper()
        == str(customer_id).strip().upper()
    ]

    if not result.empty:
        return result.iloc[0].to_dict()

    return None


# =====================================================
# 9. KNOWLEDGE BASE
# =====================================================

def search_knowledge(question):

    if knowledge.empty:
        return ""

    question = str(question).lower()

    if "question" not in knowledge.columns:
        return ""

    for _, row in knowledge.iterrows():

        stored_question = str(
            row.get("question", "")
        ).lower()

        answer = str(
            row.get("answer", "")
        )

        if stored_question and (
            stored_question in question
            or question in stored_question
        ):
            return answer

    return ""


# =====================================================
# 10. CREATE SUPPORT TICKET
# =====================================================

def create_ticket(customer_id, subject, message):

    global tickets

    new_ticket = {
        "ticket_id": "TKT-" + str(uuid.uuid4())[:8].upper(),
        "customer_id": customer_id,
        "subject": subject,
        "message": message,
        "status": "Open",
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    tickets = pd.concat(
        [
            tickets,
            pd.DataFrame([new_ticket])
        ],
        ignore_index=True
    )

    tickets.to_csv("tickets.csv", index=False)

    return new_ticket["ticket_id"]


# =====================================================
# 11. GROQ AI RESPONSE
# =====================================================

def get_ai_response(user_message):

    if client is None:
        return (
            "⚠️ Groq API key is missing. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
        )

    policy_context = ""

    if not knowledge.empty:
        policy_context = knowledge.to_string(index=False)

    prompt = f"""
You are SmartSupport AI, a friendly customer support assistant.

Your responsibilities:
- Answer customer questions.
- Explain return and refund policies.
- Explain delivery and shipping policies.
- Help customers with complaints.
- Use simple and professional English.
- Never invent order details.
- Never claim an order has been checked unless the database confirms it.
- Never reveal private information belonging to other customers.

Company Knowledge Base:
{policy_context}

Customer Message:
{user_message}

Provide a helpful and concise answer.
"""

    try:

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SmartSupport AI Customer Support Assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )

        answer = completion.choices[0].message.content

        return answer or "Sorry, I couldn't generate a response. Please try again."

    except Exception as error:

        return f"Sorry, AI service error: {str(error)}"


# =====================================================
# 12. CHATBOT RESPONSE
# =====================================================

def chatbot_response(user_message):

    user_message = str(user_message).strip()

    if not user_message:
        return "Please enter your message."

    order_id = extract_order_id(user_message)

    if order_id:

        order = search_order(order_id)

        if order:
            return format_order_response(order)

        return (
            f"⚠️ Order **{order_id}** was not found in our database.\n\n"
            "Please check your Order ID and try again."
        )

    policy_answer = search_knowledge(user_message)

    if policy_answer:
        return policy_answer

    return get_ai_response(user_message)


# =====================================================
# 13. SESSION STATE
# =====================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================
# 14. HEADER
# =====================================================

st.title("🤖 SmartSupport AI")

st.markdown("""
### Your Intelligent Customer Support Assistant

Get instant answers, track your orders, and receive
customer support through our AI-powered assistant.
""")

st.divider()


# =====================================================
# 15. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ SmartSupport Menu")

page = st.sidebar.radio(
    "Choose a Section",
    [
        "AI Customer Chat",
        "Admin Dashboard",
        "Customers",
        "Orders",
        "Support Tickets"
    ]
)


# =====================================================
# 16. CUSTOMER CHAT
# =====================================================

if page == "AI Customer Chat":

    st.subheader("💬 Chat with SmartSupport AI")

    st.info(
        "Ask questions, track orders, or learn about "
        "our return and refund policies."
    )

    # Display previous conversation

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input

    user_message = st.chat_input(
        "Example: Check my order ORD-2026-1001"
    )

    if user_message:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):

            with st.spinner("Checking your request..."):

                answer = chatbot_response(user_message)

            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []
        st.rerun()


# =====================================================
# 17. ADMIN DASHBOARD
# =====================================================

elif page == "Admin Dashboard":

    st.subheader("📊 Admin Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(customers))
    col2.metric("Total Orders", len(orders))
    col3.metric("Support Tickets", len(tickets))

    st.divider()

    if not orders.empty and "status" in orders.columns:

        st.markdown("### 📦 Order Status Overview")

        st.bar_chart(
            orders["status"].value_counts()
        )

    if not tickets.empty and "status" in tickets.columns:

        st.markdown("### 🎫 Ticket Status Overview")

        st.bar_chart(
            tickets["status"].value_counts()
        )


# =====================================================
# 18. CUSTOMER DATABASE
# =====================================================

elif page == "Customers":

    st.subheader("👥 Customer Database")

    if customers.empty:

        st.warning("No customer records found.")

    else:

        st.dataframe(
            customers,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Customers CSV",
            customers.to_csv(index=False).encode("utf-8"),
            file_name="customers_report.csv",
            mime="text/csv"
        )


# =====================================================
# 19. ORDER MANAGEMENT
# =====================================================

elif page == "Orders":

    st.subheader("📦 Order Management")

    if orders.empty:

        st.warning("No orders found.")

    else:

        st.dataframe(
            orders,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Orders CSV",
            orders.to_csv(index=False).encode("utf-8"),
            file_name="orders_report.csv",
            mime="text/csv"
        )

    st.divider()

    st.markdown("### 🔍 Test Order Search")

    test_order_id = st.text_input(
        "Enter Order ID",
        placeholder="ORD-2026-1001"
    )

    if st.button("Search Order"):

        found_order = search_order(test_order_id)

        if found_order:

            st.success("Order found successfully!")
            st.json(found_order)

        else:

            st.error(
                "Order not found. Please check your CSV database."
            )

    with st.expander("Database Diagnostic"):

        st.write("Total orders:", len(orders))

        st.write("CSV column headings:", list(orders.columns))

        if "order_id" in orders.columns:

            st.write("Available Order IDs:")
            st.write(orders["order_id"].tolist())


# =====================================================
# 20. SUPPORT TICKETS
# =====================================================

elif page == "Support Tickets":

    st.subheader("🎫 Customer Support Tickets")

    with st.form("ticket_form"):

        customer_id = st.text_input(
            "Customer ID",
            placeholder="CUS001"
        )

        subject = st.text_input(
            "Ticket Subject",
            placeholder="Delivery Issue"
        )

        message = st.text_area(
            "Describe Your Issue"
        )

        submit_ticket = st.form_submit_button(
            "Create Support Ticket"
        )

        if submit_ticket:

            if not subject or not message:

                st.error(
                    "Please enter the subject and message."
                )

            else:

                ticket_id = create_ticket(
                    customer_id,
                    subject,
                    message
                )

                st.success(
                    f"Ticket created! Your Ticket ID is {ticket_id}"
                )

    st.divider()

    st.markdown("### Existing Support Tickets")

    if tickets.empty:

        st.info("No support tickets available.")

    else:

        st.dataframe(
            tickets,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Tickets CSV",
            tickets.to_csv(index=False).encode("utf-8"),
            file_name="support_tickets_report.csv",
            mime="text/csv"
        )


# =====================================================
# 21. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:15px;
        font-size:14px;
    ">
        <b style="color:#174ea6;">SmartSupport AI</b><br>
        Powered by Groq AI + Streamlit<br>
        Intelligent Customer Support System
    </div>
    """,
    unsafe_allow_html=True
)
