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

nba_stats_writer = Agent(
    llm=llm_llama8b,
    role="NBA Stats Writer",
    goal="Write a clear, concise summary of NBA all-time leaders in a specific statistical category",
    backstory="An experienced sports writer who can translate raw stats into engaging, informative content",
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
    Analyze the user prompt to determine which NBA statistical category they're interested in.
    Use the get_nba_all_time_leaders tool to find this information.
    The user may ask about leaders in points, assists, rebounds, steals, blocks, field goal percentage, free throw percentage, or three-point percentage.
    Return the top 10 players by default, unless the user specifies a different number.
    User prompt: {user_prompt}
    ''',
    expected_output='A list of all-time leaders for the specified NBA statistical category',
    agent=nba_researcher
)

write_all_time_leaders_summary = Task(
    description='''
    Write a clear, concise summary of the NBA all-time leaders in the specified statistical category.
    Use the data provided by the get_all_time_leaders task.
    Format the information in an easy-to-read, engaging manner.
    Include the player's name, their stat value, and their rank.
    If available, include the team(s) they played for.
    Provide any interesting context or records related to these achievements.
    
    In your response, first include the raw data from get_all_time_leaders, then provide your human-readable summary.
    Separate the two parts with a line of dashes (---).
    ''',
    expected_output='Raw data followed by a human-readable summary of NBA all-time leaders in the specified statistical category',
    agent=nba_stats_writer,
    dependencies=[get_all_time_leaders],
    context=[get_all_time_leaders]
)

def route_prompt(user_prompt):
    # List of keywords that might indicate a request for all-time stats
    all_time_keywords = ['all-time', 'all time', 'leader', 'record', 'history', 'career']
    
    # Check if any of the keywords are in the user prompt
    if any(keyword in user_prompt.lower() for keyword in all_time_keywords):
        return 'player_stats'
    else:
        return 'game_info'

# Crew for game info and recap
game_info_crew = Crew(
    agents=[nba_researcher, nba_statistician, nba_writer_llama, nba_writer_gemma, nba_writer_mixtral, nba_editor],
    tasks=[
        collect_game_info, 
        retrieve_player_stats,
        write_game_recap_llama, write_game_recap_gemma, write_game_recap_mixtral,
        edit_game_recap
    ],
    verbose=False
)

# Crew for all-time player stats
player_stats_crew = Crew(
    agents=[nba_researcher, nba_stats_writer],
    tasks=[get_all_time_leaders, write_all_time_leaders_summary],
    verbose=False
)

def main():
    user_prompt = input("What would you like to know about the NBA? ")
    default_date = datetime.now().date() - timedelta(1)  # Set default date to yesterday

    route = route_prompt(user_prompt)

    if route == 'player_stats':
        result = player_stats_crew.kickoff(inputs={"user_prompt": user_prompt})
        print("Result:")
        parts = str(result).split('---')
        if len(parts) > 1:
            print("Raw Data:")
            print(parts[0].strip())
            print("\nHuman-Readable Summary:")
            print(parts[1].strip())
        else:
            print(result)
    else:  # game_info
        result = game_info_crew.kickoff(inputs={"user_prompt": user_prompt, "default_date": str(default_date)})
        print("Result:")
        print(result)

if __name__ == "__main__":
    main()