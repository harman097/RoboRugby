# RoboRugby

Training environment for ML agents in a competitive robot rugby game using OpenAI's gym standard.

## Installation

Pick a folder to store the project (ex. ~/Projects)

Download project

```sh
git clone https://github.com/harman097/RoboRugby.git
```

Install dependencies

```sh
cd /path/to/RoboRugby
python3 -m venv .venv
```

On Windows, run:
```.venv\Scripts\activate.bat```

On Unix or MacOS, run:
```source .venv/bin/activate```

```sh
pip3 install -e .
```

To deactivate the virtual environment (venv), run ```deactivate```, or close your terminal.

## Running

Activate virtual environment

```sh
cd /path/to/RoboRugby/
```

On Windows, run:
```venv\Scripts\activate.bat```

On Unix or MacOS, run:
```source venv/bin/activate```

Run game

```python3 main.py```
