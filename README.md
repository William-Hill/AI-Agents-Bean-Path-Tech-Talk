# Using AI Agent Workflows to Create Your Personal Research Team

## Overview

This project demonstrates how to use AI agent workflows to create a personal research team using the Mixture-of-Agents approach. By leveraging CrewAI, this project implements a collaborative AI framework that utilizes multiple large language models (LLMs) to enhance task performance. Additionally, the project integrates Groq's high-performance AI inference solutions to optimize the processing capabilities of these models.

## What is Groq?

Groq is a technology company specializing in high-performance AI computing. Groq provides AI inference solutions that deliver unparalleled speed and efficiency, enabling faster model execution and reduced latency. Their solutions are designed to accelerate the deployment of AI models in production environments, making them ideal for tasks requiring high computational power.

## What is CrewAI?

CrewAI is a framework that facilitates the orchestration of multiple AI agents to work collaboratively on complex tasks. It allows the integration of various LLMs to perform specific roles, such as generating, refining, and synthesizing responses. CrewAI enables the dynamic assignment of tasks to different agents, leveraging the unique strengths of each model to improve overall output quality.

## What is the Mixture-of-Agents Approach?

The Mixture-of-Agents (MoA) approach involves using multiple AI agents in a layered architecture to collaborate on a task. Each agent contributes to the task by processing inputs from previous agents and refining the output. This iterative process continues until the final output is achieved, which is typically of higher quality than what any single agent could produce. The MoA approach is particularly effective for tasks requiring diverse perspectives and expertise, such as research and content generation.

## Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```
2. Set Up a Python Virtual Environment
It is recommended to use a virtual environment to manage dependencies. You can set up a virtual environment using venv:

```bash
python3 -m venv venv
```
Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```
3. Install Dependencies
Install the required Python packages using pip from the requirements.txt file:

```bash
pip install -r requirements.txt
```
4. Obtain a Groq API Key
You need an API key from Groq to use their AI inference solutions. Visit Groq's website and sign up for an API key. Once you have the key, add it to your .env file as shown above.
5. Set Up Environment Variables
Create a .env file in the root directory of the project. This file will store your environment variables, such as API keys. The project uses the dotenv library to load these variables.

In your .env file, add the following:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

6. Running the Project
Once you have set up your environment and installed the dependencies, you can run the project. Ensure that your virtual environment is activated:

```bash
python main.py
```
This will start the process of creating your personal research team using the Mixture-of-Agents approach with CrewAI.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.