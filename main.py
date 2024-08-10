# Import packages
import datetime
from datetime import timedelta, datetime
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tools import *

# Load environment variables from .env file
load_dotenv()

# Retrieve the GROQ_API_KEY from the environment
groq_api_key = os.getenv("GROQ_API_KEY")

llm_llama70b=ChatGroq(model_name="llama3-70b-8192")
llm_llama8b=ChatGroq(model_name="llama3-8b-8192")
llm_gemma2=ChatGroq(model_name="gemma2-9b-it")
llm_mixtral = ChatGroq(model_name="mixtral-8x7b-32768")

mlb_researcher = Agent(
    llm=llm_llama70b,
    role="MLB Researcher",
    goal="Identify and return info for the MLB game related to the user prompt by returning the exact results of the get_game_info tool",
    backstory="An MLB researcher that identifies games for statisticians to analyze stats from",
    tools=[get_game_info],
    verbose=True,
    allow_delegation=False
)

mlb_statistician = Agent(
    llm=llm_llama70b,
    role="MLB Statistician",
    goal="Retrieve player batting and pitching stats for the game identified by the MLB Researcher",
    backstory="An MLB Statistician analyzing player boxscore stats for the relevant game",
    tools=[get_batting_stats, get_pitching_stats],
    verbose=True,
    allow_delegation=False
)

mlb_writer_llama = Agent(
    llm=llm_llama8b,
    role="MLB Writer",
    goal="Write a detailed game recap article using the provided game information and stats",
    backstory="An experienced and honest writer who does not make things up",
    tools=[],  # The writer does not need additional tools
    verbose=True,
    allow_delegation=False
)

mlb_writer_gemma = Agent(
    llm=llm_gemma2,
    role="MLB Writer",
    goal="Write a detailed game recap article using the provided game information and stats",
    backstory="An experienced and honest writer who does not make things up",
    tools=[],  # The writer does not need additional tools
    verbose=True,
    allow_delegation=False
)

mlb_writer_mixtral = Agent(
    llm=llm_mixtral,
    role="MLB Writer",
    goal="Write a detailed game recap article using the provided game information and stats",
    backstory="An experienced and honest writer who does not make things up",
    tools=[],  # The writer does not need additional tools
    verbose=True,
    allow_delegation=False
)

mlb_editor = Agent(
    llm=llm_llama70b,
    role="MLB Editor",
    goal="Edit multiple game recap articles to create the best final product.",
    backstory="An experienced editor that excels at taking the best parts of multiple texts to create the best final product",
    tools=[],  # The writer does not need additional tools
    verbose=True,
    allow_delegation=False
)

collect_game_info = Task(
    description='''
    Identify the correct game related to the user prompt and return game info using the get_game_info tool. 
    Unless a specific date is provided in the user prompt, use {default_date} as the game date
    User prompt: {user_prompt}
    ''',
    expected_output='High-level information of the relevant MLB game',
    agent=mlb_researcher
)

retrieve_batting_stats = Task(
    description='Retrieve ONLY boxscore batting stats for the relevant MLB game',
    expected_output='A table of batting boxscore stats',
    agent=mlb_statistician,
    dependencies=[collect_game_info],
    context=[collect_game_info]
)

retrieve_pitching_stats = Task(
    description='Retrieve ONLY boxscore pitching stats for the relevant MLB game',
    expected_output='A table of pitching boxscore stats',
    agent=mlb_statistician,
    dependencies=[collect_game_info],
    context=[collect_game_info]
)

write_game_recap_llama = Task(
    description='''
    Write a game recap article using the provided game information and stats.
    Key instructions:
    - Include things like final score, top performers and winning/losing pitcher.
    - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
    - Do not print the box score
    ''',
    expected_output='An MLB game recap article',
    agent=mlb_writer_llama,
    dependencies=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats],
    context=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats]
)

write_game_recap_gemma = Task(
    description='''
    Write a game recap article using the provided game information and stats.
    Key instructions:
    - Include things like final score, top performers and winning/losing pitcher.
    - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
    - Do not print the box score
    ''',
    expected_output='An MLB game recap article',
    agent=mlb_writer_gemma,
    dependencies=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats],
    context=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats]
)

write_game_recap_mixtral = Task(
    description='''
    Write a succinct game recap article using the provided game information and stats.
    Key instructions:
    - Structure with the following sections:
          - Introduction (game result, winning/losing pitchers, top performer on the winning team)
          - Other key performers on the winning team
          - Key performers on the losing team
          - Conclusion (including series result)
    - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
    - Do not print the box score or write out the section names
    ''',
    expected_output='An MLB game recap article',
    agent=mlb_writer_mixtral,
    dependencies=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats],
    context=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats]
)

edit_game_recap = Task(
    description='''
    You will be provided three game recap articles from multiple writers. Take the best of
    all three to output the optimal final article.
    
    Pay close attention to the original instructions:

    Key instructions:
        - Structure with the following sections:
          - Introduction (game result, winning/losing pitchers, top performer on the winning team)
          - Other key performers on the winning team
          - Key performers on the losing team
          - Conclusion (including series result)
        - Use ONLY the provided data and DO NOT make up any information, such as specific innings when events occurred, that isn't explicitly from the provided input.
        - Do not print the box score or write out the section names

    It is especially important that no false information, such as any inning or the inning in which an event occured, 
    is present in the final product. If a piece of information is present in one article and not the others, it is probably false
    ''',
    expected_output='An MLB game recap article',
    agent=mlb_editor,
    dependencies=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats],
    context=[collect_game_info, retrieve_batting_stats, retrieve_pitching_stats]
)

crew = Crew(
    agents=[mlb_researcher, mlb_statistician, mlb_writer_llama, mlb_writer_gemma, mlb_writer_mixtral, mlb_editor],
    tasks=[
        collect_game_info, 
        retrieve_batting_stats, retrieve_pitching_stats,
        write_game_recap_llama, write_game_recap_gemma, write_game_recap_mixtral,
        edit_game_recap
        ],
    verbose=False
)

user_prompt = 'write a recap of the Braves game on August 6, 2024'
default_date = datetime.now().date() - timedelta(1) # Set default date to yesterday in case no date is specified

result = crew.kickoff(inputs={"user_prompt": user_prompt, "default_date": str(default_date)})