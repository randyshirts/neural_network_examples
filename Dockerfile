ARG CONFIG_FILE
ARG USERNAME
ARG USER_GID
ARG USER_UID


FROM python:3.10

WORKDIR /app
COPY . .

# RUN chmod +x entrypoint.sh
RUN apt install tk

ARG CONFIG_FILE
# Have to make env variable for the entrypoint/command to be able to parse the arg
ENV CONFIG_FILE=${CONFIG_FILE}
ENTRYPOINT poetry run python main.py -c ${CONFIG_FILE}

# Set the user
# Create the user
ARG USERNAME
ARG USER_GID
ARG USER_UID
RUN groupadd --gid ${USER_GID} ${USERNAME}
RUN useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Tell poetry to not use a venv and to install to a certain location
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1

# Install dependencies and add the poetry location to path so our user can find it
RUN pip install poetry
ENV PATH="$PATH:$POETRY_HOME/bin"

RUN poetry install

# Set the user as the given username
USER ${USERNAME}
