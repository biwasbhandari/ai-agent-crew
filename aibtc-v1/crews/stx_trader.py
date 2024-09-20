import requests
import inspect
import streamlit as st
from crewai import Agent, Task
from crewai_tools import tool, Tool
from textwrap import dedent
from utils.crews import AIBTC_Crew, display_token_usage
from utils.scripts import get_timestamp


class STXTraderCrew(AIBTC_Crew):
    def __init__(self):
        super().__init__("STX Trader")
        self.latest_price = 0.0

    def setup_agents(self, llm):
        # Market analyst prompt updated with the analysis factors
        balance_fetcher = Agent(
            role="STX Balance Fetcher",
            goal=dedent(
                """
                Fetch and report the current balance of STX tokens for a given address.
                """
            ),
            tools=[AgentTools.get_address_balance_detailed],
            backstory=dedent(
                """
                You are a specialized agent responsible for retrieving accurate balance information for STX wallets. Your role is crucial in providing up-to-date financial data for decision-making.
                """
            ),
            verbose=True,
            llm=llm,
        )
        self.add_agent(balance_fetcher)

        result = AgentFunctions.get_latest_price()
        print(result)
        market_analyst = Agent(
            role="STX Market Analyst",
            goal=dedent(
                f"""
                Based on the current balance of the given user's address,
                Analyze {result.get('name')}({result.get('symbol')}) current market data and recommend a trading action based on the following metrics:
                - Current Stacks Prize= {result.get('stx_price')} usd Current prize of Stack coin\n
                - Market Cap= {result.get('market_cap')} (The total market value of a cryptocurrency's circulating supply. It indicates the free-float capitalization in the stock market.)
                - Volume (24h)= {result.get('volume_24h')} (The amount of STX traded in the last 24 hours.
                - Volume Change (24h Percentage): Indicator of liquidity.)
                - Volume/Market Cap (24h)= {result.get('volume_change_24h')} (Indicates liquidity. A higher ratio means the cryptocurrency is more liquid and easier to trade.)
                - Percentage Change (1h)= {result.get('percent_change_1h')} (The percentage change in STX price over the past hour.)
                - Percentage Change (24h)= {result.get('percent_change_24h')} (The percentage change in STX price over the past 24 hours.)
                - Percentage Change (7d)= {result.get('percent_change_7d')} (The percentage change in STX price over the past 7 days.)
                - Percentage Change (30d)= {result.get('percent_change_30d')} (The percentage change in STX price over the past 30 days.)
                - Market Cap Dominance: {result.get('market_cap_dominance')} (Indicates the relative size and influence of STX in the broader cryptocurrency market.)
                - Fully Diluted Market Cap:{result.get('fully_diluted_market_cap')} ( The total market capitalization if all STX tokens were issued.)

                Use this data to recommend a buy, sell, or hold(no action) action.
                """
            ),
            # tools= [
            #     AgentTools.get_latest_price,  # Fetches the necessary market data
            # ],
            backstory=dedent(
                """
                You are a seasoned cryptocurrency analyst specializing in Stacks (STX) tokens. Your expertise lies in interpreting short-term price movements and market trends to make informed trading recommendations.
                """
            ),
            verbose=True,
            llm=llm,
        )
        self.add_agent(market_analyst)

    def setup_tasks(self, address):
        # Market analysis task
        analyze_market_task = Task(
            description=dedent(
                """
                Analyze the current STX market conditions based on the following factors:
                - Market Cap
                - Volume (24h)
                - Volume Change (24h Percentage)
                - Volume/Market Cap ratio
                - Percentage changes over 1 hour, 24 hours, 7 days, and 30 days.
                - Market Cap Dominance
                - Fully Diluted Market Cap.

                Provide a recommendation (buy, sell, or hold) based on these metrics.
                """
            ),
            expected_output="A market analysis report with buy/sell/hold recommendation based on the provided metrics.",
            agent=self.agents[0],  # market_analyst
        )
        self.add_task(analyze_market_task)

        # Balance fetch task
        retrieve_balance_task = Task(
            description=dedent(
                f"""
                Fetch and report the current balance of STX tokens for the user's address:

                **Address:** {address}
                """),
            expected_output="The current STX balance and associated token holdings.",
            agent=self.agents[1],  # balance_fetcher
        )
        self.add_task(retrieve_balance_task)

    @staticmethod
    def get_task_inputs():
        return ["address"]

    @classmethod
    def get_all_tools(cls):
        return AgentTools.get_all_tools()


def render_crew(self):
    st.subheader("STX Trader Crew")
    st.markdown(
        "This tool will analyze STX market conditions, execute trades, and fetch wallet balances.")

    with st.form("stx_trader_form"):
        address = st.text_input(
            "STX Address", help="Enter the STX wallet address")
        submitted = st.form_submit_button("Run Analysis")

    if submitted and address:
        try:
            st.subheader("Analysis Progress")

            llm = st.session_state.llm

            # Empty containers for task and step progress
            st.session_state.crew_step_container = st.empty()
            st.session_state.crew_task_container = st.empty()

            st.session_state.crew_step_callback = []
            st.session_state.crew_task_callback = []

            # Initialize and set up the agents and tasks for the crew
            st.session_state.crew = STXTraderCrew()
            st.session_state.crew.setup_agents(llm)
            st.session_state.crew.setup_tasks(address)

            st.write("Running crew...")

            with st.spinner("Analyzing..."):
                # Execute the tasks and get results
                result = st.session_state.crew.create_crew().kickoff()

            st.success("Analysis complete!")

            # Display token usage (if applicable)
            display_token_usage(result.token_usage)

            # Get the outputs from the tasks (market recommendation and balance data)
            st.subheader("Analysis Results")
            for task_result in result.raw:
                # Display market recommendation
                if "buy" in task_result["output"].lower() or "sell" in task_result["output"].lower() or "hold" in task_result["output"].lower():
                    st.markdown(
                        f"**Market Recommendation**: {task_result['output']}")

                # Display wallet balance
                elif "stx_balance" in task_result["output"]:
                    st.subheader("STX Balance and Token Holdings")
                    balance_data = task_result['output']
                    st.write(
                        f"STX Balance: {balance_data['stx_balance']:.6f} STX")

                    # Display NFT holdings
                    if balance_data['nft_holdings']:
                        st.subheader("NFT Holdings")
                        for nft, details in balance_data['nft_holdings'].items():
                            st.write(f"{nft}: {details['count']} owned")

                    # Display fungible token holdings
                    if balance_data['fungible_tokens']:
                        st.subheader("Fungible Token Holdings")
                        for token, details in balance_data['fungible_tokens'].items():
                            st.write(
                                f"{token}: {int(details['balance']) / 1_000_000:.6f}")

            # Option to download the report
            timestamp = get_timestamp()
            st.download_button(
                label="Download Analysis Report (Text)",
                data=str(result.raw),
                file_name=f"{timestamp}_stx_trader_analysis.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your inputs and try again.")
    else:
        st.write("Enter STX Address, then click 'Run Analysis' to see results.")

# helper function


class AgentFunctions:
    def get_latest_price():
        """Fetch the latest STX token price from CoinMarketCap API."""
        try:
            reqUrl = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?symbol=STX"
            headersList = {
                'X-CMC_PRO_API_KEY': '4af07e42-b968-46e9-8694-1b54b6ec881e'
            }
            response = requests.request("GET", reqUrl, headers=headersList)
            stx_data = response.json()
            print(stx_data, 'stx_data')
            return {
                "name": stx_data["data"]["STX"]["name"],
                "symbol": stx_data["data"]["STX"]["symbol"],
                "stx_price": stx_data["data"]["STX"]["quote"]["USD"]["price"],
                "market_cap": stx_data["data"]["STX"]["quote"]["USD"]["market_cap"],
                "volume_24h": stx_data["data"]["STX"]["quote"]["USD"]["volume_24h"],
                "volume_change_24h": stx_data["data"]["STX"]["quote"]["USD"]["volume_change_24h"],
                "percent_change_1h": stx_data["data"]["STX"]["quote"]["USD"]["percent_change_1h"],
                "percent_change_24h": stx_data["data"]["STX"]["quote"]["USD"]["percent_change_24h"],
                "percent_change_7d": stx_data["data"]["STX"]["quote"]["USD"]["percent_change_7d"],
                "percent_change_30d": stx_data["data"]["STX"]["quote"]["USD"]["percent_change_30d"],
                "market_cap_dominance": stx_data["data"]["STX"]["quote"]["USD"]["market_cap_dominance"],
                "fully_diluted_market_cap": stx_data["data"]["STX"]["quote"]["USD"]["fully_diluted_market_cap"],
                "last_updated": stx_data["data"]["STX"]["quote"]["USD"]["last_updated"],
                "stx_circulating_supply": stx_data["data"]["STX"]["circulating_supply"]
            }
        except Exception as e:
            st.error(f"Error fetching latest price: {str(e)}")
            return {}

#########################
# Agent Tools
#########################


class AgentTools:

    @staticmethod
    @tool("Get Address Balance Detailed")
    def get_address_balance_detailed(address: str):
        """Fetch detailed balance information for the specified Stacks address using Hiro API."""
        try:
            url = f"https://api.hiro.so/extended/v1/address/{address}/balances"
            response = requests.get(url)

            if response.status_code == 200:
                balance_data = response.json()
                stx_balance = balance_data.get("stx", {}).get("balance", 0)
                nft_holdings = balance_data.get("non_fungible_tokens", {})
                fungible_holdings = balance_data.get("fungible_tokens", {})

                return {
                    # Convert to STX from microSTX
                    "stx_balance": int(stx_balance) / 1_000_000,
                    "nft_holdings": nft_holdings,
                    "fungible_tokens": fungible_holdings
                }
            else:
                st.error(
                    f"Error fetching balance: {response.status_code} {response.text}")
                return {}
        except Exception as e:
            st.error(f"Error fetching balance: {str(e)}")
            return {}

    @classmethod
    def get_all_tools(cls):
        members = inspect.getmembers(cls)
        tools = [
            member
            for name, member in members
            if isinstance(member, Tool)
            or (hasattr(member, "__wrapped__") and isinstance(member.__wrapped__, Tool))
        ]
        return tools
