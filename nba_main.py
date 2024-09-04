# Import packages
import datetime
from datetime import timedelta, datetime
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from nba_tools import get_nba_game_info, get_nba_player_stats, get_live_game_data, get_nba_all_time_leaders
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ_API_KEY from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

llm_llama70b = ChatGroq(model_name="llama3-70b-8192")
llm_llama8b = ChatGroq(model_name="llama3-8b-8192")
llm_gemma2 = ChatGroq(model_name="gemma2-9b-it")
llm_mixtral = ChatGroq(model_name="mixtral-8x7b-32768")

nba_researcher = Agent(
    llm=llm_llama70b,
    role="NBA Researcher",
    goal="Identify and return info for NBA games and all-time statistics",
    backstory="An NBA researcher that identifies games for statisticians to analyze stats from and can also provide historical statistical information",
    tools=[get_nba_game_info, get_nba_all_time_leaders],
    verbose=True,
    allow_delegation=False
)

nba_statistician = Agent(
    llm=llm_llama70b,
    role="NBA Statistician",
    goal="Retrieve player stats for the game identified by the NBA Researcher",
    backstory="An NBA Statistician analyzing player boxscore stats for the relevant game",
    tools=[get_nba_player_stats],
    verbose=True,
    allow_delegation=False
)

nba_writer_llama = Agent(
    llm=llm_llama8b,
    role="NBA Writer",
    goal="Write a detailed game recap article using the provided game information and stats",
    backstory="An experienced and honest writer who does not make things up",
    tools=[],
    verbose=True,
    allow_delegation=False
)

nba_writer_gemma = Agent(
    llm=llm_gemma2,
    role="NBA Writer",
    goal="Write a detailed game recap article using the provided game information and stats",
    backstory="An experienced and honest writer who does not make things up",
    tools=[],
    verbose=True,
    allow_delegation=False
)

nba_writer_mixtral = Agent(
    llm=llm_mixtral,
    role="NBA Writer",
    goal="Write a detailed game recap article using the provided game information and stats",
    backstory="An experienced and honest writer who does not make things up",
    tools=[],
    verbose=True,
    allow_delegation=False
)

nba_editor = Agent(
    llm=llm_llama70b,
    role="NBA Editor",
    goal="Edit multiple game recap articles to create the best final product.",
    backstory="An experienced editor that excels at taking the best parts of multiple texts to create the best final product",
    tools=[],
    verbose=True,
    allow_delegation=False
)

collect_game_info = Task(
    description='''
    Identify the correct NBA game related to the user prompt and return game info using the get_nba_game_info tool. 
    Unless a specific date is provided in the user prompt, use {default_date} as the game date.
    If no game is found, inform the user and suggest trying a different date or team.
    User prompt: {user_prompt}
    ''',
    expected_output='High-level information of the relevant NBA game or an error message if no game is found',
    agent=nba_researcher
)

retrieve_player_stats = Task(
    description='Retrieve ONLY boxscore player stats for the relevant NBA game',
    expected_output='A table of player boxscore stats',
    agent=nba_statistician,
    dependencies=[collect_game_info],
    context=[collect_game_info]
)

write_game_recap_llama = Task(
    description='''
    Write a game recap article using the provided NBA game information and stats.
    Key instructions:
    - Include things like final score, top scorers, and key team stats.
    - Use ONLY the provided data and DO NOT make up any information, such as specific quarters when events occurred, that isn't explicitly from the provided input.
    - Do not print the box score
    ''',
    expected_output='An NBA game recap article',
    agent=nba_writer_llama,
    dependencies=[collect_game_info, retrieve_player_stats],
    context=[collect_game_info, retrieve_player_stats]
)

write_game_recap_gemma = Task(
    description='''
    Write a game recap article using the provided NBA game information and stats.
    Key instructions:
    - Include things like final score, top scorers, and key team stats.
    - Use ONLY the provided data and DO NOT make up any information, such as specific quarters when events occurred, that isn't explicitly from the provided input.
    - Do not print the box score
    ''',
    expected_output='An NBA game recap article',
    agent=nba_writer_gemma,
    dependencies=[collect_game_info, retrieve_player_stats],
    context=[collect_game_info, retrieve_player_stats]
)

write_game_recap_mixtral = Task(
    description='''
    Write a succinct NBA game recap article using the provided game information and stats.
    Key instructions:
    - Structure with the following sections:
          - Introduction (game result, top scorer, key team stats)
          - Key performers on the winning team
          - Key performers on the losing team
          - Conclusion (including series result if applicable)
    - Use ONLY the provided data and DO NOT make up any information, such as specific quarters when events occurred, that isn't explicitly from the provided input.
    - Do not print the box score or write out the section names
    ''',
    expected_output='An NBA game recap article',
    agent=nba_writer_mixtral,
    dependencies=[collect_game_info, retrieve_player_stats],
    context=[collect_game_info, retrieve_player_stats]
)

edit_game_recap = Task(
    description='''
    You will be provided three game recap articles from multiple writers. Take the best of
    all three to output the optimal final article.
    
    Pay close attention to the original instructions:

    Key instructions:
        - Structure with the following sections:
          - Introduction (game result, top scorer, key team stats)
          - Key performers on the winning team
          - Key performers on the losing team
          - Conclusion (including series result if applicable)
        - Use ONLY the provided data and DO NOT make up any information, such as specific quarters when events occurred, that isn't explicitly from the provided input.
        - Do not print the box score or write out the section names

    It is especially important that no false information, such as any quarter or the quarter in which an event occurred, 
    is present in the final product. If a piece of information is present in one article and not the others, it is probably false
    ''',
    expected_output='An NBA game recap article',
    agent=nba_editor,
    dependencies=[write_game_recap_llama, write_game_recap_gemma, write_game_recap_mixtral],
    context=[collect_game_info, retrieve_player_stats, write_game_recap_llama, write_game_recap_gemma, write_game_recap_mixtral]
)

get_all_time_leaders = Task(
    description='''
    Retrieve the all-time leaders for a specific NBA statistical category.
    Use the get_nba_all_time_leaders tool to find this information.
    The user may ask for leaders in points, assists, rebounds, steals, blocks, field goal percentage, free throw percentage, or three-point percentage.
    Return the top 10 players by default, unless the user specifies a different number.
    User prompt: {user_prompt}
    ''',
    expected_output='A list of all-time leaders for the specified NBA statistical category',
    agent=nba_researcher
)

crew = Crew(
    agents=[nba_researcher, nba_statistician, nba_writer_llama, nba_writer_gemma, nba_writer_mixtral, nba_editor],
    tasks=[
        collect_game_info, 
        retrieve_player_stats,
        get_all_time_leaders,  # Add this new task
        write_game_recap_llama, write_game_recap_gemma, write_game_recap_mixtral,
        edit_game_recap
    ],
    verbose=False
)

user_prompt = input("Which NBA game would you like to be recapped? ")
default_date = datetime.now().date() - timedelta(1)  # Set default date to yesterday in case no date is specified

result = crew.kickoff(inputs={"user_prompt": user_prompt, "default_date": str(default_date)})
print(result)