from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict
from IPython.display import Image, display
from dotenv import load_dotenv
_ = load_dotenv()

## IMPORT PER AGENT ##
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

## IMPORT PER GRAFO ##
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

## IMPORT PER LLM ##
from langchain.chat_models import init_chat_model

SYSTEM_PROMPT = """"""
USER_PROMPT = """"""

## Sezione per la definizione llm ##
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

def get_llm():
    llm = init_chat_model("deepseek/deepseek-chat:free",base_url ="https://openrouter.ai/api/v1")
    # llm.with_structured_output(Router)
    return llm

## Sezione per la definizione dell'agent e delle funzioni di supporto ##
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

tools = [write_email]

def create_prompt(state):
    prompt_instructions = {"agent_instructions": "Please respond to the email."}
    agent_system_prompt = "Hello, I am Gigi. I am a Gigi agent. {instructions}"
    return [
        {
            "role": "system", 
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                )
        }
    ] + state['messages']

def get_agent(llm,tools:list):
    
    return create_react_agent(
        llm,
        tools=tools,
        prompt="Hello, I am Gigi. I am a Gigi agent. Please respond to the email.",
    )

def get_response(agent, query:str):
    return agent.invoke(
                {"messages": [{
                    "role": "user", 
                    "content": query
                }]}
             )["messages"][-1].pretty_print()


## Sezione per la definizione del grafo ##

class State(TypedDict):
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
]:

    system_prompt = SYSTEM_PROMPT.format(
        examples=None
    )
    user_prompt = USER_PROMPT.format()
    result = get_llm().invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

def get_graph_agent(agent):
    email_agent = StateGraph(State)
    email_agent = email_agent.add_node(triage_router)
    email_agent = email_agent.add_node("response_agent", agent)
    email_agent = email_agent.add_edge(START, "triage_router")
    email_agent = email_agent.compile()

def print_graph_status(agent):
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))



def main():
    email_input = {
    "author": "Marketing Team <marketing@amazingdeals.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "🔥 EXCLUSIVE OFFER: Limited Time Discount on Developer Tools! 🔥",
    "email_thread": """Dear Valued Developer,

    Don't miss out on this INCREDIBLE opportunity! 

    🚀 For a LIMITED TIME ONLY, get 80% OFF on our Premium Developer Suite! 

    ✨ FEATURES:
    - Revolutionary AI-powered code completion
    - Cloud-based development environment
    - 24/7 customer support
    - And much more!

    💰 Regular Price: $999/month
    🎉 YOUR SPECIAL PRICE: Just $199/month!

    🕒 Hurry! This offer expires in:
    24 HOURS ONLY!

    Click here to claim your discount: https://amazingdeals.com/special-offer

    Best regards,
    Marketing Team
    ---
    To unsubscribe, click here
    """,
    }
    
    # Istanzio llm
    llm = get_llm()

    # Istanzio agent
    agent = get_agent(llm,tools)
    agent.invoke(
    {"messages": [{
        "role": "user", 
        "content": "what is my availability for tuesday?"
    }]}
)
    # Istanzio grafo
    email_agent = get_graph_agent(agent)

    response = email_agent.invoke({"email_input": email_input})
    for m in response["messages"]:
        m.pretty_print()

main()