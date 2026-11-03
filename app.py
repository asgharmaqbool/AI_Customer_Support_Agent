
import os
import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq


# ==========================================
# 1. APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartSupport AI",
    page_icon="🤖",
    layout="wide"
)

MODEL = "openai/gpt-oss-120b"

st.markdown("""
<style>
.main {
    background-color: #f5f8ff;
}
h1, h2, h3 {
    color: #174ea6;
}
.stButton button {
    background-color: #1769e0;
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
[data-testid="stSidebar"] {
    background-color: #eaf1ff;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOAD GROQ API KEY
# ==========================================

def get_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


api_key = get_api_key()

client = Groq(api_key=api_key) if api_key else None


# ==========================================
# 3. CSV DATABASE FILES
# ==========================================

def load_data(filename, columns):
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename, dtype=str).fillna("")
        except Exception:
            st.error(f"Unable to read {filename}. Please check the CSV file.")
            return pd.DataFrame(columns=columns)

    return pd.DataFrame(columns=columns)


customers = load_data(
    "customers.csv",
    ["customer_id", "name", "email", "phone"]
)

orders = load_data(
    "orders.csv",
    ["order_id", "customer_id", "product", "order_date",
     "status", "amount", "tracking_number"]
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


# ==========================================
# 4. ORDER ID MATCHING - FIXED
# ==========================================

def normalize_order_id(order_id):
    """
    Removes spaces and hyphens so different formats
    can match the same order ID.
    Example:
    ORD-2026-1002 = ORD20261002
    """

    return re.sub(r"[^A-Z0-9]", "", str(order_id).upper())


def extract_order_id(message):
    """
    Recognizes IDs such as:
    ORD-2026-1002
    ORD20261002
    ORD-1002
    """

    pattern = r"\bORD(?:[-\s]?\d{4})?[-\s]?\d{3,}\b"

    match = re.search(pattern, str(message).upper())

    if match:
        return match.group(0).strip()

    return None


def search_order(order_id):
    """
    Searches the orders.csv database using a normalized ID.
    """

    if orders.empty or "order_id" not in orders.columns:
        return None

    searched_id = normalize_order_id(order_id)

    for _, order in orders.iterrows():
        database_id = normalize_order_id(order.get("order_id", ""))

        if database_id == searched_id:
            return order.to_dict()

    return None


# ==========================================
# 5. CUSTOMER LOOKUP
# ==========================================

def find_customer(customer_id):
    if customers.empty or "customer_id" not in customers.columns:
        return None

    result = customers[
        customers["customer_id"].astype(str).str.strip().str.upper()
        == str(customer_id).strip().upper()
    ]

    if not result.empty:
        return result.iloc[0].to_dict()

    return None


# ==========================================
# 6. ORDER RESPONSE
# ==========================================

def format_order_response(order):
    """
    Shows basic order information.
    Does not reveal private contact details.
    """

    response = "### 📦 Your Order Details\n\n"

    display_fields = {
        "order_id": "Order ID",
        "product": "Product",
        "order_date": "Order Date",
        "status": "Order Status",
        "amount": "Order Amount",
        "tracking_number": "Tracking Number",
        "expected_delivery": "Expected Delivery"
    }

    for field, label in display_fields.items():
        value = order.get(field, "")

        if value and str(value).strip():
            response += f"**{label}:** {value}\n\n"

    response += (
        "Is there anything else I can help you with?"
    )

    return response


# ==========================================
# 7. KNOWLEDGE BASE SEARCH
# ==========================================

def search_knowledge(question):
    if knowledge.empty:
        return ""

    question = str(question).lower()

    if "question" not in knowledge.columns:
        return ""

    for _, row in knowledge.iterrows():
        stored_question = str(row.get("question", "")).lower()
        answer = str(row.get("answer", ""))

        if stored_question and (
            stored_question in question
            or question in stored_question
        ):
            return answer

    return ""


# ==========================================
# 8. CREATE SUPPORT TICKET
# ==========================================

def create_ticket(customer_id, subject, message):

    global tickets

    new_ticket = {
        "ticket_id": "TKT-" + str(uuid.uuid4())[:8].upper(),
        "customer_id": customer_id,
        "subject": subject,
        "message": message,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    tickets = pd.concat(
        [tickets, pd.DataFrame([new_ticket])],
        ignore_index=True
    )

    tickets.to_csv("tickets.csv", index=False)

    return new_ticket["ticket_id"]


# ==========================================
# 9. GROQ AI RESPONSE
# ==========================================

def get_ai_response(user_message):

    if client is None:
        return (
            "⚠️ Groq API key is missing. "
            "Please configure GROQ_API_KEY in Streamlit Secrets."
        )

    policy_context = ""

    if not knowledge.empty:
        policy_context = knowledge.to_string(index=False)

    prompt = f"""
You are SmartSupport AI, a professional customer support assistant.

Your responsibilities:
1. Help customers with their questions.
2. Explain return and refund policies.
3. Explain delivery and shipping policies.
4. Help customers with complaints.
5. Use simple, friendly English.
6. Never invent order details.
7. Never claim an order has been found unless the database confirms it.
8. Never reveal another customer's personal information.

Company Knowledge Base:
{policy_context}

Customer Message:
{user_message}

Give a helpful and concise response.
If the customer wants order tracking but has not provided an order ID,
ask them to provide their Order ID.
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are SmartSupport AI Customer Support Assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Sorry, AI service error: {str(e)}"


# ==========================================
# 10. CHATBOT RESPONSE LOGIC
# ==========================================

def chatbot_response(user_message):

    user_message = str(user_message).strip()

    if not user_message:
        return "Please enter your message."

    # IMPORTANT: Check order IDs BEFORE calling Groq AI.

    order_id = extract_order_id(user_message)

    if order_id:

        order = search_order(order_id)

        if order:
            return format_order_response(order)

        return (
            f"⚠️ I could not find order **{order_id}** "
            "in our order database.\n\n"
            "Please check that you entered the correct Order ID. "
            "You can also contact our support team for assistance."
        )

    # Check knowledge base
    policy_answer = search_knowledge(user_message)

    if policy_answer:
        return policy_answer

    # If no order ID or policy answer, use AI
    return get_ai_response(user_message)


# ==========================================
# 11. SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================
# 12. HEADER
# ==========================================

st.title("🤖 SmartSupport AI")

st.markdown(
    """
    ### Your Intelligent Customer Support Assistant

    Get instant answers, track your orders, and receive
    support with our AI-powered customer service system.
    """
)

st.divider()


# ==========================================
# 13. SIDEBAR NAVIGATION
# ==========================================

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


# ==========================================
# 14. CUSTOMER CHAT
# ==========================================

if page == "AI Customer Chat":

    st.subheader("💬 Chat with SmartSupport AI")

    st.info(
        "You can ask questions, check an order, "
        "or learn about our refund and return policies."
    )

    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input(
        "Example: Please check my order ORD-2026-1002"
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

            with st.spinner("SmartSupport AI is checking..."):

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


# ==========================================
# 15. ADMIN DASHBOARD
# ==========================================

elif page == "Admin Dashboard":

    st.subheader("📊 Admin Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(customers))
    col2.metric("Total Orders", len(orders))
    col3.metric("Support Tickets", len(tickets))

    st.divider()

    if not orders.empty and "status" in orders.columns:

        st.markdown("### 📦 Order Status Overview")

        status_counts = orders["status"].value_counts()

        st.bar_chart(status_counts)

    if not tickets.empty and "status" in tickets.columns:

        st.markdown("### 🎫 Ticket Status Overview")

        ticket_counts = tickets["status"].value_counts()

        st.bar_chart(ticket_counts)


# ==========================================
# 16. CUSTOMERS
# ==========================================

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


# ==========================================
# 17. ORDERS
# ==========================================

elif page == "Orders":

    st.subheader("📦 Order Management")

    if orders.empty:
        st.warning("No orders found in orders.csv.")

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
            placeholder="ORD-2026-1002"
        )

        if st.button("Search Order"):

            found_order = search_order(test_order_id)

            if found_order:
                st.success("Order found successfully!")

                st.json(found_order)

            else:
                st.error("Order ID not found. Please check your CSV database.")


# ==========================================
# 18. SUPPORT TICKETS
# ==========================================

elif page == "Support Tickets":

    st.subheader("🎫 Customer Support Tickets")

    with st.form("ticket_form"):

        customer_id = st.text_input(
            "Customer ID",
            placeholder="CUS001"
        )

        subject = st.text_input(
            "Ticket Subject",
            placeholder="Delivery issue"
        )

        message = st.text_area(
            "Describe Your Issue"
        )

        submit_ticket = st.form_submit_button(
            "Create Support Ticket"
        )

        if submit_ticket:

            if not subject or not message:
                st.error("Please enter both subject and message.")

            else:

                ticket_id = create_ticket(
                    customer_id,
                    subject,
                    message
                )

                st.success(
                    f"Ticket created successfully! Ticket ID: {ticket_id}"
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


# ==========================================
# 19. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        <b>SmartSupport AI</b><br>
        Powered by Groq AI + Streamlit<br>
        Intelligent Customer Support System
    </div>
    """,
    unsafe_allow_html=True
)
